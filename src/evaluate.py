"""Paired, held-out generation for base and adapter, plus safety review samples."""

import argparse
import json
import re

from src.common import load_config, project_path, write_json
from src.train import SYSTEM


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def scores(prediction, reference):
    a, b = normalize(prediction).split(), normalize(reference).split()
    common = sum(min(a.count(word), b.count(word)) for word in set(a) & set(b))
    precision = common / len(a) if a else 0.0
    recall = common / len(b) if b else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"normalized_exact_match": float(normalize(prediction) == normalize(reference)), "token_f1": f1}


def aggregate(rows):
    if not rows:
        raise ValueError("No evaluation rows")
    return {key: sum(row["scores"][key] for row in rows) / len(rows)
            for key in ("normalized_exact_match", "token_f1")}


def generate(model, tokenizer, question, max_new_tokens):
    import torch
    prompt = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    encoded = tokenizer.apply_chat_template(prompt, add_generation_prompt=True,
                                            tokenize=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0, encoded.shape[-1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    training_dir = project_path(config["training"]["output_dir"])
    metadata = json.loads((training_dir / "metadata.json").read_text(encoding="utf-8"))
    dataset_dir = project_path(config["dataset"]["processed_dir"])
    tests = [json.loads(line) for line in (dataset_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    tests = tests[:config["evaluation"]["max_test_samples"]]
    if not tests:
        raise ValueError("Held-out test split is empty")
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a CUDA GPU")
    precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=precision)
    tokenizer = AutoTokenizer.from_pretrained(training_dir / "adapter")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(config["model"]["name"], quantization_config=quant,
                                               device_map={"": 0}, torch_dtype=precision)
    model = PeftModel.from_pretrained(base, str(training_dir / "adapter"))
    model.eval()
    paired = []
    for row in tests:
        with model.disable_adapter():
            base_answer = generate(model, tokenizer, row["instruction"], config["evaluation"]["max_new_tokens"])
        tuned_answer = generate(model, tokenizer, row["instruction"], config["evaluation"]["max_new_tokens"])
        paired.append({"question": row["instruction"], "reference": row["output"],
                       "source_id": row.get("source_id"),
                       "base": {"answer": base_answer, "scores": scores(base_answer, row["output"])},
                       "tuned": {"answer": tuned_answer, "scores": scores(tuned_answer, row["output"])}})
    cases = json.loads(project_path("data/evaluation/safety.json").read_text(encoding="utf-8"))
    safety = []
    for case in cases:
        with model.disable_adapter():
            base_answer = generate(model, tokenizer, case["question"], config["evaluation"]["max_new_tokens"])
        tuned_answer = generate(model, tokenizer, case["question"], config["evaluation"]["max_new_tokens"])
        # Flags are prompts for human review, not a medical-safety score.
        flags = [pattern for pattern in case["flag_patterns"] if re.search(pattern, tuned_answer, re.I)]
        safety.append({"id": case["id"], "category": case["category"], "question": case["question"],
                       "base_answer": base_answer, "tuned_answer": tuned_answer, "flags": flags})
    report = {"base_model": config["model"]["name"], "adapter": str(training_dir / "adapter"),
              "dataset_sha256": metadata["dataset"]["archive_sha256"], "test_size": len(paired),
              "base_metrics": aggregate([row["base"] for row in paired]),
              "tuned_metrics": aggregate([row["tuned"] for row in paired]),
              "safety_cases": safety,
              "limitations": "Reference overlap cannot establish correctness or clinical safety. Human review is required."}
    reports = project_path("artifacts/reports")
    write_json(reports / "evaluation.json", report)
    write_json(reports / "predictions.json", paired)
    review = reports / "safety_review.json"
    if not review.exists():
        write_json(review, {case["id"]: {"safe": None, "notes": ""} for case in cases})


if __name__ == "__main__":
    main()
