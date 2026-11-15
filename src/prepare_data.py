"""Download the upstream MedQuAD XML and make leakage-safe JSONL splits."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import statistics
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import zipfile

from src.common import load_config, project_path, setup_logging, write_json

LOG = logging.getLogger(__name__)
# These three folders have empty answers because upstream removed copyrighted text.
EXCLUDED_FOLDERS = {"10_MPlus_ADAM_QA", "11_MPlusDrugs_QA", "12_MPlusHerbsSupplements_QA"}


def clean_text(value):
    return re.sub(r"\s+", " ", value or "").strip()


def download_source(url, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "MedQuAD.zip"
    if not archive.exists():
        LOG.info("Downloading upstream MedQuAD archive")
        partial = raw_dir / "MedQuAD.zip.partial"
        try:
            with urlopen(url, timeout=120) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            with zipfile.ZipFile(partial) as zipped:
                if not zipped.namelist():
                    raise ValueError("Downloaded MedQuAD archive is empty")
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return archive, digest


def records_from_archive(archive):
    """Read XML directly from ZIP; do not alter or extract the raw archive."""
    with zipfile.ZipFile(archive) as zipped:
        members = sorted(name for name in zipped.namelist() if name.lower().endswith(".xml"))
        if not members:
            raise ValueError("No XML records in MedQuAD archive")
        for member in members:
            parts = Path(member).parts
            if any(part in EXCLUDED_FOLDERS for part in parts):
                continue
            try:
                root = ET.fromstring(zipped.read(member))
            except ET.ParseError as exc:
                LOG.warning("Malformed XML in %s: %s", member, exc)
                yield {"instruction": "", "output": "", "source_file": member, "source_url": ""}
                continue
            source_url = root.get("url", "")
            for pair in root.findall(".//QAPair"):
                question, answer = pair.find("Question"), pair.find("Answer")
                yield {
                    "instruction": clean_text("".join(question.itertext()) if question is not None else ""),
                    "output": clean_text("".join(answer.itertext()) if answer is not None else ""),
                    "source_file": member,
                    "source_url": source_url,
                    "source_id": question.get("qid", "") if question is not None else "",
                }


def prepare(records, config):
    ds = config["dataset"]
    counts = Counter()
    unique = {}
    for record in records:
        counts["discovered"] += 1
        question, answer = clean_text(record.get("instruction")), clean_text(record.get("output"))
        if not question or not answer:
            counts["missing_values"] += 1
            counts["invalid"] += 1
            continue
        if len(question) > ds["max_question_chars"] or len(answer) > ds["max_answer_chars"]:
            counts["too_long"] += 1
            counts["invalid"] += 1
            continue  # Truncating a medical answer could change its meaning.
        key = (question.casefold(), answer.casefold())
        if key in unique:
            counts["duplicates"] += 1
            continue
        unique[key] = {
            "instruction": question, "input": "", "output": answer,
            "source_file": record.get("source_file", ""),
            "source_url": record.get("source_url", ""),
            "source_id": record.get("source_id", ""),
        }
    if not unique:
        raise ValueError("No valid question-answer records found")
    groups = defaultdict(list)
    for record in unique.values():
        groups[record["instruction"].casefold()].append(record)
    keys = sorted(groups)
    random.Random(ds["seed"]).shuffle(keys)
    if len(keys) < 3:
        raise ValueError("At least three distinct questions are required for three splits")
    train_end = max(1, min(len(keys) - 2, round(len(keys) * ds["train_fraction"])))
    val_end = max(train_end + 1, min(len(keys) - 1, round(len(keys) * (ds["train_fraction"] + ds["validation_fraction"]))))
    splits = {
        "train": [row for key in keys[:train_end] for row in groups[key]],
        "validation": [row for key in keys[train_end:val_end] for row in groups[key]],
        "test": [row for key in keys[val_end:] for row in groups[key]],
    }
    counts["valid"] = sum(map(len, splits.values()))
    counts["removed"] = counts["invalid"] + counts["duplicates"]
    lengths = {
        field: {"min": min(values), "median": statistics.median(values), "max": max(values)}
        for field, values in (
            (field, [len(row[field]) for row in unique.values()])
            for field in ("instruction", "output")
        )
    }
    report = {"counts": dict(counts), "splits": {name: len(rows) for name, rows in splits.items()},
              "distinct_questions": len(keys), "length_characters": lengths, "seed": ds["seed"]}
    return splits, report


def save_splits(splits, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        with (destination / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--archive", type=Path, help="Use an already downloaded upstream ZIP")
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    if args.archive:
        archive = args.archive
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    else:
        archive, digest = download_source(config["dataset"]["archive_url"], project_path(config["dataset"]["raw_dir"]))
    splits, report = prepare(records_from_archive(archive), config)
    report["source"] = config["dataset"]["source"]
    report["archive_sha256"] = digest
    destination = project_path(config["dataset"]["processed_dir"])
    save_splits(splits, destination)
    write_json(destination / "quality_report.json", report)
    LOG.info("Prepared %s records: %s", report["counts"]["valid"], report["splits"])


if __name__ == "__main__":
    main()
