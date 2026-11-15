#!/usr/bin/env bash
set -euo pipefail

# Update only the previously configured private code dataset.
: "${KAGGLE_USERNAME:?KAGGLE_USERNAME is required}"
: "${KAGGLE_KEY:?KAGGLE_KEY is required}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage="$root/kaggle/_stage"
mkdir -p "$stage/src" "$stage/configs" "$stage/data/evaluation"
cp "$root"/src/*.py "$stage/src/"
cp "$root/configs/config.yaml" "$stage/configs/config.yaml"
cp "$root/data/evaluation/safety.json" "$stage/data/evaluation/safety.json"
cp "$root/requirements.txt" "$root/requirements-ci.txt" "$stage/"
kaggle datasets metadata "$KAGGLE_USERNAME/llama-finetuning-code" -p "$stage"
tar -cf "$stage/project.tar" -C "$stage" src configs data requirements.txt requirements-ci.txt
kaggle datasets version -p "$stage" -m "Update medical fine-tuning code" --dir-mode skip
