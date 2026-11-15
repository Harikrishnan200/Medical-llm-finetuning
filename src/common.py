"""Small shared helpers: configuration, paths and JSON output."""

import json
import logging
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs/config.yaml"
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for section in ("dataset", "model", "training", "evaluation", "publishing"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    ds, tr = config["dataset"], config["training"]
    if not 0 < ds["train_fraction"] < 1 or not 0 < ds["validation_fraction"] < 1:
        raise ValueError("Split fractions must be between zero and one")
    if ds["train_fraction"] + ds["validation_fraction"] >= 1:
        raise ValueError("Train and validation fractions must leave a test split")
    for key in ("batch_size", "gradient_accumulation_steps", "max_length", "lora_rank"):
        if tr[key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    if tr["eval_strategy"] != tr["save_strategy"]:
        raise ValueError("Best checkpoint loading requires matching eval/save strategies")
    if tr["quantization"] != "nf4":
        raise ValueError("Only NF4 QLoRA is supported")
    return config


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
