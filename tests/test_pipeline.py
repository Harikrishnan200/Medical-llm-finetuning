import copy
import sys
from types import SimpleNamespace

import pytest

from src.common import load_config
from src.evaluate import scores, aggregate
from src.report import promotion_gate
from src.upload_adapter import can_publish
from src.analyze_training import analyze
from src.train import choose_precision, gpu_info


def test_config_validation(tmp_path):
    config = load_config()
    assert config["training"]["lora_rank"] > 0
    path = tmp_path / "bad.yaml"
    path.write_text("dataset: {}\n")
    with pytest.raises(ValueError, match="section"):
        load_config(path)


def test_metrics_and_analysis():
    assert scores("The heart.", "the heart")["token_f1"] == 1
    assert scores("unrelated", "heart")["token_f1"] == 0
    assert aggregate([{"scores": scores("heart", "heart")}])["token_f1"] == 1
    result = analyze([{"step": 1, "loss": 0.2}, {"step": 1, "eval_loss": 0.9}],
                     {"duration_seconds": 10}, 0.5)
    assert result["warnings"]


def test_promotion_and_publishing():
    config = load_config()
    evaluation = {"base_metrics": {"token_f1": 0.2}, "tuned_metrics": {"token_f1": 0.3},
                  "safety_cases": [{"id": "one", "flags": []}]}
    assert promotion_gate(evaluation, {"one": {"safe": None}}, config)[0] == "PASS WITH WARNINGS"
    assert promotion_gate(evaluation, {"one": {"safe": True}}, config)[0] == "PASS"
    assert promotion_gate(evaluation, {"one": {"safe": False}}, config)[0] == "FAIL"
    bad = copy.deepcopy(evaluation)
    bad["tuned_metrics"]["token_f1"] = 0.1
    assert promotion_gate(bad, {}, config)[0] == "FAIL"
    assert not can_publish(config, {"status": "PASS"})
    config["publishing"].update(enabled=True, repo_id="owner/adapter")
    assert can_publish(config, {"status": "PASS"})
    assert not can_publish(config, {"status": "PASS WITH WARNINGS"})


def test_t4_uses_fp16_and_multi_gpu_is_rejected(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                           get_device_capability=lambda index: (7, 5),
                           is_bf16_supported=lambda: True)
    fake_torch = SimpleNamespace(cuda=cuda)
    assert choose_precision(fake_torch) == "fp16"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(RuntimeError, match="one visible GPU"):
        gpu_info(14)
    cuda.device_count = lambda: 1
    cuda.get_device_capability = lambda index: (8, 0)
    assert choose_precision(fake_torch) == "bf16"
