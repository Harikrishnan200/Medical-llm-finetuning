"""Build Kaggle metadata from the current account and queue a notebook run."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def metadata(username):
    return {"id": f"{username}/medical-medquad-qlora", "title": "Medical MedQuAD QLoRA",
            "code_file": "train.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [f"{username}/llama-finetuning-code"],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        raise RuntimeError("KAGGLE_USERNAME is required")
    stage = ROOT / "_launch"
    stage.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "train.ipynb", stage / "train.ipynb")
    (stage / "kernel-metadata.json").write_text(json.dumps(metadata(username), indent=2), encoding="utf-8")
    if args.dry_run:
        print(f"Validated metadata for {username}/medical-medquad-qlora; no GPU job queued")
        return
    if not os.environ.get("KAGGLE_KEY"):
        raise RuntimeError("KAGGLE_KEY is required")
    subprocess.run(["kaggle", "kernels", "push", "-p", str(stage)], check=True)
    print(f"Submission accepted or queued: https://www.kaggle.com/code/{username}/medical-medquad-qlora")
    print("Check actual completion with: kaggle kernels status " + username + "/medical-medquad-qlora")
    print("Submission is not proof that training completed.")


if __name__ == "__main__":
    main()
