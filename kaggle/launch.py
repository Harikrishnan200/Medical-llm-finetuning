"""Build Kaggle metadata from the current account and queue a notebook run."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def metadata(username):
    return {"id": f"{username}/medical-medquad-qlora", "title": "Medical MedQuAD QLoRA",
            "code_file": "train.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [f"{username}/llama-finetuning-code"],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}


def package_code(archive_path):
    """Stage only the files required by the GPU notebook."""
    files = sorted((PROJECT / "src").glob("*.py")) + [
        PROJECT / "configs/config.yaml",
        PROJECT / "data/evaluation/safety.json",
        PROJECT / "requirements.txt",
        PROJECT / "requirements-ci.txt",
    ]
    with tarfile.open(archive_path, "w") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(PROJECT).as_posix())
    return [path.relative_to(PROJECT).as_posix() for path in files]


def stage_dataset(username, dry_run=False):
    if dry_run:
        stage = ROOT / "_stage"
        stage.mkdir(exist_ok=True)
        files = package_code(stage / "project.tar")
        print("Local package validated:", ", ".join(files))
        print("No Kaggle dataset was changed")
        return
    if not os.environ.get("KAGGLE_KEY"):
        raise RuntimeError("KAGGLE_KEY is required")
    with tempfile.TemporaryDirectory(prefix="medquad-stage-", dir=ROOT) as temporary:
        stage = Path(temporary)
        package_code(stage / "project.tar")
        dataset_id = f"{username}/llama-finetuning-code"
        subprocess.run(["kaggle", "datasets", "metadata", dataset_id, "-p", str(stage)], check=True)
        subprocess.run(["kaggle", "datasets", "version", "-p", str(stage), "-m",
                        "Update medical fine-tuning code", "--dir-mode", "skip"], check=True)
        print("Staged code in private Kaggle dataset:", dataset_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage", action="store_true", help="Version the existing private code dataset")
    parser.add_argument("--stage-dry-run", action="store_true", help="Package code locally without Kaggle access")
    args = parser.parse_args()
    load_dotenv(PROJECT / ".env", override=False)
    username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        raise RuntimeError("KAGGLE_USERNAME is required")
    if args.stage or args.stage_dry_run:
        stage_dataset(username, dry_run=args.stage_dry_run)
        return
    stage = ROOT / "_launch"
    stage.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "train.ipynb", stage / "train.ipynb")
    (stage / "kernel-metadata.json").write_text(json.dumps(metadata(username), indent=2), encoding="utf-8")
    if args.dry_run:
        print("Validated notebook metadata for configured Kaggle account; no GPU job queued")
        return
    if not os.environ.get("KAGGLE_KEY"):
        raise RuntimeError("KAGGLE_KEY is required")
    subprocess.run(["kaggle", "kernels", "push", "-p", str(stage)], check=True)
    print(f"Submission accepted or queued: https://www.kaggle.com/code/{username}/medical-medquad-qlora")
    print("Check actual completion with: kaggle kernels status " + username + "/medical-medquad-qlora")
    print("Submission is not proof that training completed.")


if __name__ == "__main__":
    main()
