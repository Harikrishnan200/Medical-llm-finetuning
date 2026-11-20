"""Optional gated upload of adapter files only."""

import argparse
import json
import os

from src.common import load_config, project_path


def can_publish(config, promotion):
    return (config["publishing"]["enabled"] is True
            and bool(config["publishing"]["repo_id"])
            and promotion.get("status") == "PASS")


def get_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        raise RuntimeError("HF_TOKEN is required for private upload; add it in Kaggle Add-ons > Secrets and grant this notebook access") from exc
    if not token:
        raise RuntimeError("HF_TOKEN is empty in Kaggle Secrets")
    return token


def ensure_private_repo(api, repo):
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True, private=True)
    if api.repo_info(repo_id=repo, repo_type="model").private is not True:
        raise RuntimeError(f"Refusing to upload to a non-private Hugging Face repository: {repo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    promotion = json.loads(project_path("artifacts/reports/promotion.json").read_text(encoding="utf-8"))
    if not can_publish(config, promotion):
        print("Publishing skipped: explicit enablement, repository and PASS are all required")
        return
    token = get_hf_token()
    from huggingface_hub import HfApi
    adapter = project_path(config["training"]["output_dir"]) / "adapter"
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError("Completed adapter is missing")
    model_card = ("---\nbase_model: " + config["model"]["name"] + "\n---\n\n"
                  "# Educational MedQuAD QLoRA adapter\n\n"
                  "Base model license: " + config["model"]["license"] + ". Adapter license must be chosen by its owner.\n\n"
                  "Fine-tuned on MedQuAD (CC BY 4.0), upstream https://github.com/abachaa/MedQuAD. "
                  "Original source authors: Asma Ben Abacha and Dina Demner-Fushman. "
                  "See the upstream citation and source terms. The three MedlinePlus subsets with removed answers are excluded.\n\n"
                  "This adapter is for education and research. It is not clinically validated or suitable for diagnosis, "
                  "prescribing, dosage, or emergency triage. See evaluation reports in the project repository.\n")
    (adapter / "README.md").write_text(model_card, encoding="utf-8")
    api = HfApi(token=token)
    repo = config["publishing"]["repo_id"]
    ensure_private_repo(api, repo)
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(adapter),
                      ignore_patterns=["*.bin", "checkpoint-*"])
    print(f"Published adapter to private repository {repo}")


if __name__ == "__main__":
    main()
