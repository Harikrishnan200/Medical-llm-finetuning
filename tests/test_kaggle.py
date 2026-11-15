import json
from pathlib import Path
import runpy
import tarfile

import pytest


KAGGLE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "kaggle/launch.py"))


def test_stage_package_excludes_secrets(tmp_path):
    archive = tmp_path / "project.tar"
    names = KAGGLE["package_code"](archive)
    with tarfile.open(archive) as packed:
        assert sorted(packed.getnames()) == sorted(names)
    assert "src/train.py" in names
    assert "configs/config.yaml" in names
    assert ".env" not in names
    assert not any(name.startswith("artifacts/") or name.startswith("data/raw/") for name in names)


def test_kaggle_metadata_conversion_and_private_guard():
    source = {"info": {"title": "Existing code dataset", "isPrivate": True,
                       "licenses": [{"name": "other"}], "description": "Prior description"}}
    converted = KAGGLE["upload_metadata"](source, "owner/llama-finetuning-code")
    assert converted["id"] == "owner/llama-finetuning-code"
    assert converted["licenses"] == source["info"]["licenses"]
    assert converted["description"] == "Prior description"
    assert converted["is_private"] is True
    assert "info" not in converted
    source["info"]["isPrivate"] = False
    with pytest.raises(ValueError, match="private"):
        KAGGLE["upload_metadata"](source, "owner/llama-finetuning-code")


def test_legacy_code_requires_explicit_replacement():
    with pytest.raises(RuntimeError, match="legacy"):
        KAGGLE["check_existing_files"]([{"name": "old_script.py"}], False)
    KAGGLE["check_existing_files"]([{"name": "old_script.py"}], True)
    KAGGLE["check_existing_files"]([{"name": "project.tar"}], False)
