"""Summarize actual Trainer history and plot training/validation losses."""

import argparse
import json

from src.common import load_config, project_path, write_json


def analyze(history, metadata, warning_gap):
    training = [(row.get("step", 0), row["loss"]) for row in history if "loss" in row]
    validation = [(row.get("step", 0), row["eval_loss"]) for row in history if "eval_loss" in row]
    if not training or not validation:
        raise ValueError("History has no training or validation loss")
    gap = validation[-1][1] - training[-1][1]
    return {"final_train_loss": training[-1][1], "final_validation_loss": validation[-1][1],
            "best_validation_loss": min(loss for _, loss in validation),
            "duration_seconds": metadata["duration_seconds"], "overfit_gap": gap,
            "warnings": ["Validation loss exceeds training loss by the configured gap; inspect the curves."] if gap > warning_gap else [],
            "training_points": training, "validation_points": validation}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    training_dir = project_path(config["training"]["output_dir"])
    history = json.loads((training_dir / "history.json").read_text(encoding="utf-8"))
    metadata = json.loads((training_dir / "metadata.json").read_text(encoding="utf-8"))
    result = analyze(history, metadata, config["evaluation"]["overfit_gap_warning"])
    reports, plots = project_path("artifacts/reports"), project_path("artifacts/plots")
    write_json(reports / "training_analysis.json", result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots.mkdir(parents=True, exist_ok=True)
    for key, label in (("training_points", "Training"), ("validation_points", "Validation")):
        x, y = zip(*result[key])
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel("Optimizer step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "loss_curve.png")
    plt.close()
    summary = (f"# Training summary\n\nBase: {metadata['base_model']}\n\n"
               f"Duration: {result['duration_seconds']:.0f} seconds\n\n"
               f"Best validation loss: {result['best_validation_loss']:.4f}\n\n"
               f"Final train / validation loss: {result['final_train_loss']:.4f} / {result['final_validation_loss']:.4f}\n\n"
               "A growing validation loss while training loss falls suggests overfitting. "
               "The configured gap is a heuristic, not a diagnosis.\n\n"
               f"Warnings: {result['warnings'] or 'none'}\n")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "training_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
