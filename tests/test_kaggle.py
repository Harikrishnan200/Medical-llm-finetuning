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
    KAGGLE["check_existing_files"]([{"name": name} for name in
                                    ("src/train.py", "src/prepare_data.py", "configs/config.yaml", "requirements.txt")], False)


def test_notebook_setup_accepts_expanded_kaggle_dataset(tmp_path, monkeypatch):
    notebook = json.loads((Path(__file__).resolve().parents[1] / "kaggle/train.ipynb").read_text())
    setup = "".join(next(cell["source"] for cell in notebook["cells"] if cell.get("id") == "setup"))
    input_root = tmp_path / "input" / "llama-finetuning-code"
    project = tmp_path / "working" / "medical-project"
    for name in ("src/train.py", "configs/config.yaml", "data/evaluation/safety.json", "requirements.txt"):
        path = input_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    setup = setup.replace("/kaggle/input/llama-finetuning-code", str(input_root).replace("\\", "/"))
    setup = setup.replace("/kaggle/working/medical-project", str(project).replace("\\", "/"))
    monkeypatch.chdir(tmp_path)
    exec(compile(setup, "kaggle/train.ipynb setup", "exec"), {})
    assert (project / "src/train.py").read_text() == "src/train.py"
    assert (project / "data/evaluation/safety.json").is_file()


def test_notebook_setup_accepts_archive_kaggle_dataset(tmp_path, monkeypatch):
    notebook = json.loads((Path(__file__).resolve().parents[1] / "kaggle/train.ipynb").read_text())
    setup = "".join(next(cell["source"] for cell in notebook["cells"] if cell.get("id") == "setup"))
    input_root = tmp_path / "input" / "llama-finetuning-code"
    project = tmp_path / "working" / "medical-project"
    input_root.mkdir(parents=True)
    source = tmp_path / "train.py"
    source.write_text("print('staged')")
    with tarfile.open(input_root / "project.tar", "w") as archive:
        archive.add(source, arcname="src/train.py")
    setup = setup.replace("/kaggle/input/llama-finetuning-code", str(input_root).replace("\\", "/"))
    setup = setup.replace("/kaggle/working/medical-project", str(project).replace("\\", "/"))
    monkeypatch.chdir(tmp_path)
    exec(compile(setup, "kaggle/train.ipynb setup", "exec"), {})
    assert (project / "src/train.py").read_text() == "print('staged')"


@pytest.mark.parametrize("stale_mount", [False, True])
def test_bundled_notebook_uses_submitted_code(tmp_path, monkeypatch, stale_mount):
    notebook = KAGGLE["bundled_notebook"]()
    bundle = "".join(next(cell["source"] for cell in notebook["cells"] if cell.get("id") == "bundled-project"))
    setup = "".join(next(cell["source"] for cell in notebook["cells"] if cell.get("id") == "setup"))
    input_root = tmp_path / "input" / "llama-finetuning-code"
    project = tmp_path / "working" / "medical-project"
    if stale_mount:
        stale_source = input_root / "src/train.py"
        stale_source.parent.mkdir(parents=True)
        stale_source.write_text("old dataset version")
    setup = setup.replace("/kaggle/input/llama-finetuning-code", str(input_root).replace("\\", "/"))
    setup = setup.replace("/kaggle/working/medical-project", str(project).replace("\\", "/"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    namespace = {}
    exec(compile(bundle, "kaggle/train.ipynb bundle", "exec"), namespace)
    exec(compile(setup, "kaggle/train.ipynb setup", "exec"), namespace)
    assert (project / "src/train.py").read_text() == (Path(__file__).resolve().parents[1] / "src/train.py").read_text()
    assert (project / "configs/config.yaml").is_file()
    assert not (project / ".env").exists()
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "0"
