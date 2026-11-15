import json
from pathlib import Path
import zipfile

import pytest

from src.common import load_config
from src.prepare_data import clean_text, prepare, records_from_archive, save_splits


def sample_records():
    return [
        {"instruction": "  What is A?  ", "output": " Answer A "},
        {"instruction": "What is A?", "output": "Answer A"},
        {"instruction": "What is A?", "output": "Another answer"},
        {"instruction": "What is B?", "output": "Answer B"},
        {"instruction": "What is C?", "output": "Answer C"},
        {"instruction": "What is D?", "output": "Answer D"},
        {"instruction": "", "output": "Missing question"},
        {"instruction": "What is E?", "output": ""},
    ]


def test_clean_split_and_quality(tmp_path):
    config = load_config()
    first, report = prepare(sample_records(), config)
    second, _ = prepare(sample_records(), config)
    assert first == second
    assert report["counts"]["discovered"] == 8
    assert report["counts"]["duplicates"] == 1
    assert report["counts"]["missing_values"] == 2
    question_sets = [{row["instruction"] for row in split} for split in first.values()]
    assert all(not a & b for index, a in enumerate(question_sets) for b in question_sets[index + 1:])
    save_splits(first, tmp_path)
    assert sum(len((tmp_path / f"{name}.jsonl").read_text().splitlines()) for name in first) == 5
    for name in first:
        for line in (tmp_path / f"{name}.jsonl").read_text().splitlines():
            assert set(("instruction", "input", "output")) <= set(json.loads(line))


def test_real_xml_shape_and_excluded_folders(tmp_path):
    archive = tmp_path / "sample.zip"
    xml = '<Document source="CDC" url="https://example.org"><QAPairs><QAPair><Question qid="1">Q?</Question><Answer>A.</Answer></QAPair></QAPairs></Document>'
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("MedQuAD/9_CDC_QA/example.xml", xml)
        zipped.writestr("MedQuAD/10_MPlus_ADAM_QA/example.xml", xml)
    records = list(records_from_archive(archive))
    assert len(records) == 1
    assert records[0]["source_id"] == "1"
    assert clean_text(" a\n b ") == "a b"


def test_no_valid_records():
    with pytest.raises(ValueError, match="No valid"):
        prepare([{"instruction": "", "output": ""}], load_config())
