"""Promotion decision and readable evaluation report."""

import argparse
import json

from src.common import load_config, project_path, write_json


def promotion_gate(evaluation, review, config):
    gain = evaluation["tuned_metrics"]["token_f1"] - evaluation["base_metrics"]["token_f1"]
    flags = sum(bool(row["flags"]) for row in evaluation["safety_cases"])
    if gain < config["evaluation"]["min_f1_gain"] or flags > config["evaluation"]["max_safety_flags"]:
        return "FAIL", f"F1 gain {gain:.4f}; flagged safety cases {flags}"
    decisions = [review.get(row["id"], {}).get("safe") for row in evaluation["safety_cases"]]
    if any(value is False for value in decisions):
        return "FAIL", "Human review marked at least one safety case unsafe"
    if not all(value is True for value in decisions):
        return "PASS WITH WARNINGS", "Human safety review is incomplete"
    return "PASS", "Configured reference and safety checks passed; no clinical validation implied"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    directory = project_path("artifacts/reports")
    evaluation = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    review = json.loads((directory / "safety_review.json").read_text(encoding="utf-8"))
    status, reason = promotion_gate(evaluation, review, config)
    write_json(directory / "promotion.json", {"status": status, "reason": reason})
    text = (f"# Evaluation report\n\nBase model: {evaluation['base_model']}\n\n"
            f"Dataset archive SHA-256: `{evaluation['dataset_sha256']}`\n\n"
            f"Held-out examples: {evaluation['test_size']}\n\n"
            "| Metric | Base | Adapter | Difference |\n|---|---:|---:|---:|\n")
    for key in ("normalized_exact_match", "token_f1"):
        base, tuned = evaluation["base_metrics"][key], evaluation["tuned_metrics"][key]
        text += f"| {key} | {base:.4f} | {tuned:.4f} | {tuned-base:+.4f} |\n"
    text += (f"\nPromotion: **{status}** — {reason}\n\n"
             "Safety cases and both responses are in `evaluation.json`; review decisions are in `safety_review.json`. "
             "Inspect relevance, completeness, unsupported claims, and harmful advice manually. "
             "Token overlap and pattern flags cannot prove medical correctness or clinical safety.\n")
    (directory / "evaluation_report.md").write_text(text, encoding="utf-8")
    print(status, reason)


if __name__ == "__main__":
    main()
