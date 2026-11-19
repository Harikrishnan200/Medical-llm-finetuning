"""Build Kaggle metadata from the current account and queue a notebook run."""

import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
KAGGLE = [sys.executable, "-m", "kaggle"]


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


def bundled_notebook():
    """Include the same allowlisted source in case Kaggle omits the dataset mount."""
    notebook = json.loads((ROOT / "train.ipynb").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="medquad-bundle-", dir=ROOT) as temporary:
        archive = Path(temporary) / "project.tar"
        package_code(archive)
        encoded = base64.b64encode(archive.read_bytes()).decode("ascii")
    chunks = [encoded[index:index + 120] for index in range(0, len(encoded), 120)]
    notebook["cells"].insert(1, {
        "cell_type": "code", "id": "bundled-project", "execution_count": None,
        "metadata": {}, "outputs": [],
        "source": ["import base64\n", "BUNDLED_PROJECT_TAR = base64.b64decode(\n"]
        + [f"    '{chunk}'\n" for chunk in chunks] + [")\n"],
    })
    return notebook


def upload_metadata(downloaded, dataset_id):
    """Kaggle's download format nests metadata; uploads need a flat document."""
    info = downloaded.get("info", downloaded)
    if info.get("isPrivate", info.get("is_private")) is not True:
        raise ValueError("Refusing to version a dataset that is not confirmed private")
    if not info.get("title") or not info.get("licenses"):
        raise ValueError("Existing Kaggle dataset metadata lacks title or license")
    result = {"id": dataset_id, "title": info["title"],
              "licenses": info["licenses"], "is_private": True}
    for field in ("subtitle", "description", "expectedUpdateFrequency"):
        if info.get(field):
            result[field] = info[field]
    return result


def check_existing_files(files, allow_replace):
    names = {item["name"] for item in files}
    expanded = {"src/train.py", "src/prepare_data.py", "configs/config.yaml", "requirements.txt"}
    if names and "project.tar" not in names and not expanded.issubset(names) and not allow_replace:
        raise RuntimeError("Existing Kaggle dataset contains legacy files. Review them and rerun with --replace-existing-code to version this project over them.")


def stage_dataset(username, dry_run=False, allow_replace=False):
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
        subprocess.run([*KAGGLE, "datasets", "metadata", dataset_id, "-p", str(stage)], check=True)
        metadata_path = stage / "dataset-metadata.json"
        downloaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_path.write_text(json.dumps(upload_metadata(downloaded, dataset_id), indent=2), encoding="utf-8")
        files = subprocess.run([*KAGGLE, "datasets", "files", dataset_id, "--format", "json",
                                "--page-size", "200"], check=True, capture_output=True, text=True)
        check_existing_files(json.loads(files.stdout), allow_replace)
        subprocess.run([*KAGGLE, "datasets", "version", "-p", str(stage), "-m",
                        "Update medical fine-tuning code", "--dir-mode", "skip"], check=True)
        print("Staged code in private Kaggle dataset:", dataset_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage", action="store_true", help="Version the existing private code dataset")
    parser.add_argument("--stage-dry-run", action="store_true", help="Package code locally without Kaggle access")
    parser.add_argument("--replace-existing-code", action="store_true", help="Allow replacing legacy files in the existing private code dataset")
    args = parser.parse_args()
    load_dotenv(PROJECT / ".env", override=False)
    username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        raise RuntimeError("KAGGLE_USERNAME is required")
    if args.stage or args.stage_dry_run:
        stage_dataset(username, dry_run=args.stage_dry_run, allow_replace=args.replace_existing_code)
        return
    stage = ROOT / "_launch"
    stage.mkdir(exist_ok=True)
    (stage / "train.ipynb").write_text(json.dumps(bundled_notebook(), indent=1), encoding="utf-8")
    (stage / "kernel-metadata.json").write_text(json.dumps(metadata(username), indent=2), encoding="utf-8")
    if args.dry_run:
        print("Validated notebook metadata for configured Kaggle account; no GPU job queued")
        return
    if not os.environ.get("KAGGLE_KEY"):
        raise RuntimeError("KAGGLE_KEY is required")
    subprocess.run([*KAGGLE, "kernels", "push", "-p", str(stage)], check=True)
    print(f"Submission accepted or queued: https://www.kaggle.com/code/{username}/medical-medquad-qlora")
    print("Check actual completion with: kaggle kernels status " + username + "/medical-medquad-qlora")
    print("Submission is not proof that training completed.")


if __name__ == "__main__":
    main()
