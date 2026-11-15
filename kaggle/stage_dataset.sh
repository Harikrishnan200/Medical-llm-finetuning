#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${KAGGLE_REPLACE_EXISTING_CODE:-false}" == "true" ]]; then
  python "$root/kaggle/launch.py" --stage --replace-existing-code
else
  python "$root/kaggle/launch.py" --stage
fi
