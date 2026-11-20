"""Single-GPU QLoRA supervised fine-tuning. Run on Kaggle with a CUDA GPU."""

import argparse
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import subprocess
import time

from src.common import load_config, project_path, setup_logging, write_json

LOG = logging.getLogger(__name__)
SYSTEM = "Provide general medical education. Do not diagnose, prescribe, or give personalized dosing. For emergencies, advise immediate local emergency care."


def choose_precision(torch):
    """Use BF16 only when the GPU supports it natively (T4 uses FP16)."""
    major, _ = torch.cuda.get_device_capability(0)
    return "bf16" if major >= 8 and torch.cuda.is_bf16_supported() else "fp16"


def training_arguments(config, precision):
    tr = config["training"]
    from trl import SFTConfig
    return SFTConfig(
        output_dir=str(project_path(tr["output_dir"])),
        num_train_epochs=tr["epochs"], learning_rate=tr["learning_rate"],
        per_device_train_batch_size=tr["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=tr["gradient_accumulation_steps"],
        max_length=tr["max_length"], warmup_ratio=tr["warmup_ratio"],
        weight_decay=tr["weight_decay"], eval_strategy=tr["eval_strategy"],
        save_strategy=tr["save_strategy"], logging_steps=tr["logging_steps"],
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        gradient_checkpointing=tr["gradient_checkpointing"],
        fp16=precision == "fp16", bf16=precision == "bf16",
        report_to=["tensorboard"], seed=tr["seed"],
        packing=False, completion_only_loss=True,
    )


def conversational_row(row):
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": row["instruction"]}],
        "completion": [{"role": "assistant", "content": row["output"]}],
    }


def gpu_info(min_gb):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for 4-bit QLoRA. Enable a Kaggle GPU accelerator.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("QLoRA requires one visible GPU. Set CUDA_VISIBLE_DEVICES=0 before starting Python.")
    props = torch.cuda.get_device_properties(0)
    memory_gb = props.total_memory / 1024**3
    if memory_gb < min_gb:
        raise RuntimeError(f"GPU has {memory_gb:.1f} GiB; configuration requires {min_gb} GiB")
    precision = choose_precision(torch)
    info = {"name": props.name, "memory_gib": round(memory_gb, 2),
            "cuda": torch.version.cuda, "torch": torch.__version__, "precision": precision}
    LOG.info("GPU: %s", info)
    return info


def log_training_run(metadata, training_config):
    """Record a completed adapter run without making tracking a training gate."""
    try:
        import mlflow
        database = project_path("artifacts/training/mlflow.db").resolve()
        mlflow.set_tracking_uri("sqlite:///" + database.as_posix())
        mlflow.set_experiment("medical-medquad-qlora")
        with mlflow.start_run():
            mlflow.log_params({"base_model": metadata["base_model"],
                               "dataset_sha256": metadata["dataset"]["archive_sha256"],
                               "git_commit": metadata["git_commit"] or "unknown", **training_config})
            mlflow.log_metrics({"train_loss": metadata["train_loss"],
                                "validation_loss": metadata["validation_loss"],
                                "duration_seconds": metadata["duration_seconds"]})
            mlflow.set_tag("adapter_path", metadata["adapter"])
    except Exception as exc:
        LOG.warning("MLflow tracking failed; adapter and JSON metadata remain available: %s", exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              EarlyStoppingCallback)
    from trl import SFTTrainer

    tr = config["training"]
    output = project_path(tr["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    info = gpu_info(config["model"]["min_gpu_gb"])
    data_dir = project_path(config["dataset"]["processed_dir"])
    for name in ("train", "validation"):
        if not (data_dir / f"{name}.jsonl").is_file():
            raise FileNotFoundError(f"Prepare data first: missing {name}.jsonl")
    dataset = load_dataset("json", data_files={name: str(data_dir / f"{name}.jsonl") for name in ("train", "validation")})
    for name in dataset:
        dataset[name] = dataset[name].map(conversational_row, remove_columns=dataset[name].column_names)
    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = torch.bfloat16 if info["precision"] == "bf16" else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=tr["quantization"],
        bnb_4bit_use_double_quant=tr["double_quantization"],
        bnb_4bit_compute_dtype=compute_dtype,
    )
    LOG.info("Loading %s with 4-bit %s quantization", model_name, tr["quantization"])
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization, device_map={"": 0}, torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=tr["gradient_checkpointing"])
    lora = LoraConfig(r=tr["lora_rank"], lora_alpha=tr["lora_alpha"],
                      lora_dropout=tr["lora_dropout"], target_modules=tr["target_modules"],
                      bias="none", task_type="CAUSAL_LM")
    callbacks = [EarlyStoppingCallback(early_stopping_patience=tr["early_stopping_patience"])]
    trainer = SFTTrainer(model=model, args=training_arguments(config, info["precision"]),
                         train_dataset=dataset["train"], eval_dataset=dataset["validation"],
                         processing_class=tokenizer, peft_config=lora, callbacks=callbacks)
    started = time.time()
    try:
        LOG.info("Starting supervised fine-tuning")
        result = trainer.train()
        validation = trainer.evaluate()
        adapter = output / "adapter"
        trainer.save_model(str(adapter))
        tokenizer.save_pretrained(str(adapter))
        duration = time.time() - started
        metadata = {
            "base_model": model_name, "adapter": str(adapter), "duration_seconds": duration,
            "train_loss": result.training_loss, "validation_loss": validation.get("eval_loss"),
            "gpu": info, "dataset": json.loads((data_dir / "quality_report.json").read_text(encoding="utf-8")),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip() or None,
            "library_versions": {name: version(name) for name in
                                 ("torch", "transformers", "datasets", "trl", "peft", "accelerate", "bitsandbytes")},
        }
        write_json(output / "metadata.json", metadata)
        write_json(output / "config.json", config)
        log_training_run(metadata, tr)
        LOG.info("Adapter saved at %s", adapter)
    finally:
        # Keep partial history when training fails; do not write success metadata.
        write_json(output / "history.json", trainer.state.log_history)


if __name__ == "__main__":
    main()
