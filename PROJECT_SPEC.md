# PROJECT: MEDICAL LLM FINE-TUNING PIPELINE WITH QLoRA, KAGGLE GPU AND GITHUB ACTIONS

## ROLE

Act as a Senior AI Engineer, LLM Engineer, Machine Learning Engineer and MLOps Engineer.

Help me build a practical, production-style medical LLM fine-tuning pipeline using Python, Hugging Face, PyTorch, QLoRA, Kaggle GPU and GitHub Actions.

I am learning LLM engineering and MLOps. I want to understand every component of the project, not just receive generated code.

**The most important requirement is to keep the codebase simple, understandable, beginner-friendly and maintainable.**

Do not overengineer the project. Do not create unnecessary abstractions, complicated class hierarchies, excessive modules or enterprise-level infrastructure.

Build a small project that follows good production engineering practices.

---

# 1. PROJECT OBJECTIVE

Build a complete medical question-answering LLM fine-tuning pipeline using the MedQuAD dataset.

The project should cover:

1. Medical dataset collection and inspection.
2. Dataset cleaning and preprocessing.
3. Supervised Fine-Tuning (SFT).
4. LoRA / QLoRA.
5. Hugging Face Transformers.
6. PyTorch.
7. Kaggle GPU training.
8. Training analysis.
9. Base model vs fine-tuned model evaluation.
10. Model evaluation reports.
11. Experiment tracking.
12. Training monitoring and logging.
13. Docker support for reproducible local development and testing.
14. GitHub Actions CI/CD.
15. Kaggle CLI integration.
16. Optional Hugging Face Hub model publishing.
17. Documentation and reproducibility.

## Scope restriction

For this phase, build ONLY the model fine-tuning pipeline.

Do NOT build:

- FastAPI backend.
- Flask backend.
- Gradio or Streamlit application.
- React frontend.
- Chatbot interface.
- RAG pipeline.
- Vector database.
- Embedding retrieval system.
- Inference API.
- Application deployment.
- Kubernetes infrastructure.
- A separate web application.

The goal is to fine-tune, evaluate, track and manage a medical LLM adapter.

Keep the project focused on the complete ML development and training workflow.

---

# 2. MOST IMPORTANT REQUIREMENT: SIMPLE CODEBASE

I do not want an unnecessarily overstructured codebase.

Follow these rules throughout the project:

1. Keep the folder structure small.
2. Use simple Python scripts.
3. Prefer straightforward functions over unnecessary classes.
4. Avoid creating separate modules for every small operation.
5. Avoid unnecessary design patterns and abstractions.
6. Avoid introducing multiple frameworks that solve the same problem.
7. Keep configurations in one understandable configuration file where practical.
8. Use meaningful variable and function names.
9. Write comments explaining important ML concepts and non-obvious decisions.
10. Keep the training logic easy to follow from beginning to end.
11. Avoid duplicate implementations.
12. Do not create files that are not genuinely needed.

Every Python script should have a clear responsibility.

A beginner should be able to understand:

- Where the dataset comes from.
- How the dataset is prepared.
- How the model is loaded.
- How QLoRA works in the implementation.
- How training starts.
- Where the adapter is saved.
- How evaluation works.
- How results are tracked.
- How the Kaggle and GitHub automation works.

## Production-style practices

Simple does not mean poorly engineered.

Include:

- Proper logging.
- Error handling.
- Configuration management.
- Reproducible random seeds.
- Input and dataset validation.
- Clear dependency management.
- Secure secret handling.
- Unit tests.
- Training and evaluation reports.
- Git commit tracking.
- Model and dataset version information.
- Useful documentation.

Use production-style practices without introducing unnecessary complexity.

---

# 3. FIRST TASK: INSPECT MY EXISTING WORKSPACE

Before modifying anything, inspect my existing project.

I already have a previous LLM fine-tuning project with Kaggle and GitHub Actions integration.

Inspect:

1. Existing repository structure.
2. README.md.
3. Existing Python scripts.
4. Configuration files.
5. Requirements files.
6. Kaggle notebook.
7. Kaggle kernel metadata.
8. Kaggle dataset configuration.
9. Kaggle staging scripts.
10. Kaggle launch/status scripts, if present.
11. GitHub Actions workflows.
12. Existing tests.
13. Existing environment variables and secrets references.
14. Existing Docker configuration, if any.
15. Existing MLflow or TensorBoard configuration, if any.

Understand the existing architecture before making changes.

## Existing architecture to preserve

My previous implementation uses:

- GitHub repository.
- GitHub Actions.
- CPU-side validation.
- pytest.
- Kaggle CLI.
- Private Kaggle Dataset for staging code and required files.
- Kaggle GPU notebook.
- QLoRA training.
- Training analysis.
- Model evaluation.
- PASS / PASS WITH WARNINGS / FAIL promotion gate.
- Optional Hugging Face Hub adapter publishing.

The existing private Kaggle Dataset is named:

`llama-finetuning-code`

The existing staging script is:

`kaggle/stage_dataset.sh`

Existing GitHub secrets:

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- Optional `HF_TOKEN`

Kaggle may also contain an optional `HF_TOKEN` secret for model publishing.

The Kaggle notebook uses the staged private dataset through `dataset_sources`.

## Important reuse rules

- Reuse the existing Kaggle and GitHub integrations wherever possible.
- Do not unnecessarily redesign the working CI/CD architecture.
- Do not ask me to reconnect Kaggle or GitHub if the required configuration is already present.
- Do not overwrite working workflows without understanding them.
- Do not reuse unrelated personal data from the previous project.
- Reuse only appropriate infrastructure and automation code.
- Never expose secrets in logs.
- If the current workspace does not contain the previous project files, ask me to provide the relevant repository or files instead of inventing their contents.

---

# 4. DATASET: MEDQUAD

Use MedQuAD as the primary medical question-answering dataset.

Potential sources:

Original upstream repository:

https://github.com/abachaa/MedQuAD

Hugging Face dataset mirror:

https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset

Before using the dataset:

1. Inspect its Hugging Face dataset card.
2. Inspect the original upstream repository.
3. Verify the actual dataset files.
4. Inspect the actual column names and data format.
5. Verify dataset provenance.
6. Check the applicable license and usage restrictions.
7. Determine whether the processed mirror's license covers the upstream records.
8. Document any licensing limitations.

Do not assume that publicly downloadable data is automatically free for every type of use.

If licensing is unclear or does not permit the intended use, stop and explain the issue before proceeding.

Do not hardcode assumed row counts, column names or dataset statistics.

Discover and validate the actual dataset structure.

## Dataset pipeline

Implement the following:

1. Download the dataset.
2. Load the dataset.
3. Inspect the dataset.
4. Validate the dataset.
5. Handle missing values.
6. Remove malformed records.
7. Detect duplicate records.
8. Clean the question and answer text.
9. Apply reasonable length limits.
10. Split into training, validation and test datasets.
11. Generate dataset statistics.
12. Generate a dataset quality report.

Keep the raw dataset unchanged.

Use a reproducible random seed.

Avoid train/test leakage.

Where appropriate, group identical or duplicate questions before splitting so that the same question does not appear in multiple splits.

Create a data quality report showing:

- Total records discovered.
- Valid records.
- Invalid records.
- Duplicate records.
- Records removed.
- Final training records.
- Final validation records.
- Final test records.
- Missing values.
- Question and answer length statistics.

Save the processed dataset in a consistent format such as JSONL.

Do not commit large datasets or model weights to GitHub.

---

# 5. DATA PREPARATION FOR SUPERVISED FINE-TUNING

Prepare MedQuAD for instruction-based supervised fine-tuning.

Use a consistent format.

Example:

{
  "instruction": "What is hypertension?",
  "input": "",
  "output": "Hypertension is a medical condition involving persistently elevated blood pressure."
}

This is only an illustrative example. Do not insert it as a real dataset record.

Requirements:

- Preserve the original meaning of medical answers.
- Do not introduce unsupported medical information.
- Remove empty or malformed records.
- Remove unnecessary duplicates.
- Handle missing values.
- Apply appropriate text length limits.
- Validate the final JSONL structure.
- Keep training, validation and test data separate.
- Generate a dataset summary.

Explain why each preprocessing step is required.

Do not use an LLM to rewrite all the medical answers unless there is a clearly justified and separately approved reason.

Keep the original source records traceable where possible.

---

# 6. BASE MODEL SELECTION

Choose a small instruction-tuned open-weight language model suitable for QLoRA fine-tuning on Kaggle's free GPU resources.

Investigate suitable models before choosing.

Consider:

- Model size.
- Model architecture.
- Hugging Face availability.
- License and usage restrictions.
- Instruction-tuning capability.
- T4 GPU compatibility.
- VRAM requirements.
- Quantization compatibility.
- Transformers and PEFT compatibility.
- Training time.

Do not blindly choose a large model.

The previous project used the Llama 3.2 3B family, so consider whether it is a suitable candidate for this project. Also compare other practical small models if useful.

Do not assume that the previous model is automatically the best choice for medical fine-tuning.

Explain:

1. Why the selected model is appropriate.
2. Its license.
3. Its memory requirements.
4. Whether it can run on a Kaggle T4.
5. Any access restrictions.
6. Any limitations.

Make the base model configurable through the configuration file.

Do not start training until the model choice and hardware compatibility have been documented.

---

# 7. SFT AND QLoRA FINE-TUNING

Implement supervised fine-tuning using:

- Python.
- PyTorch.
- Hugging Face Transformers.
- Hugging Face Datasets.
- PEFT.
- TRL.
- bitsandbytes, where compatible.
- Accelerate, where required.

Use LoRA or QLoRA to reduce GPU memory requirements.

Prefer QLoRA if the selected model and Kaggle GPU support it reliably.

## Training requirements

Implement:

1. Tokenizer loading.
2. Dataset tokenization.
3. Base model loading.
4. GPU detection.
5. Quantization configuration.
6. LoRA configuration.
7. Training arguments.
8. Supervised fine-tuning.
9. Validation.
10. Checkpoint saving.
11. Training loss tracking.
12. Validation loss tracking.
13. Best checkpoint selection.
14. Early stopping where appropriate.
15. Final adapter saving.

Use a beginner-friendly training script.

Keep the main training flow readable.

## Configurable parameters

Store training parameters in a configuration file.

Include:

- Base model.
- Dataset paths.
- Output directory.
- Learning rate.
- Per-device batch size.
- Gradient accumulation steps.
- Number of epochs.
- Maximum sequence length.
- Warmup ratio.
- Weight decay.
- LoRA rank.
- LoRA alpha.
- LoRA dropout.
- Target modules.
- Quantization settings.
- Gradient checkpointing.
- Random seed.
- Evaluation strategy.
- Save strategy.
- Logging configuration.

Do not hardcode these settings throughout multiple scripts.

## Hardware compatibility

Detect the available GPU.

Display:

- GPU name.
- GPU memory.
- CUDA availability.
- PyTorch CUDA version.
- Selected precision.
- Quantization configuration.

Do not assume that every Kaggle session provides the same GPU.

Use hardware-compatible precision.

For example, do not blindly enable bfloat16 on a T4 GPU that does not support it.

Use float16 where appropriate.

Fail with a useful error message if the available hardware cannot support the selected configuration.

Avoid unnecessary multi-GPU complexity. Use a simple, reliable configuration appropriate for the available hardware.

## Training artifacts

Save:

- Final LoRA adapter.
- Training configuration.
- Training logs.
- Training history.
- Checkpoints where appropriate.
- Evaluation outputs.
- Training analysis.
- Model metadata.

Do not commit model weights or large training artifacts to GitHub.

---

# 8. TRAINING ANALYSIS

Implement a simple training analysis script.

Generate:

- Training loss graph.
- Validation loss graph.
- Training duration.
- Best validation loss.
- Final training loss.
- Final validation loss.
- Dataset statistics.
- Training configuration summary.
- Overfitting indicators.
- Training warnings.

Use actual training history.

Do not invent training results or metrics.

Use simple Python visualization tools such as Matplotlib.

Save outputs under:

artifacts/
    reports/
    plots/
    training/

Include:

- `training_analysis.json`
- `training_summary.md`
- `loss_curve.png`

Explain how to interpret the training and validation loss curves.

Treat overfitting thresholds as configurable heuristics, not mathematical proof.

---

# 9. MODEL EVALUATION

Build an evaluation pipeline that compares:

1. Original base model.
2. Fine-tuned model.

Use a held-out test dataset.

Both models must receive the same prompts and use equivalent generation settings.

Evaluate:

- Answer relevance.
- Answer correctness using suitable reference-based or human-review methods.
- Instruction following.
- Response completeness.
- Hallucination indicators.
- Medical safety.
- Unsupported claims.
- Response consistency.

Use suitable metrics.

Possible metrics include:

- Exact Match.
- Normalized Exact Match.
- Token-level F1.
- ROUGE, where appropriate.
- Other simple reference-based metrics when justified.

Do not rely on one automated metric to establish medical correctness.

Create a separate small safety evaluation set containing examples such as:

- General health education questions.
- Diagnosis requests.
- Prescription requests.
- Personalized dosage questions.
- Emergency-related questions.
- Questions with insufficient information.
- Questions outside the dataset's coverage.
- Attempts to bypass safety instructions.

Evaluate the model's responses.

Do not claim that passing these checks makes the model clinically safe.

## Evaluation report

Generate a report comparing the base and fine-tuned models.

Include:

- Model names.
- Dataset version.
- Evaluation dataset size.
- Evaluation metrics.
- Base model results.
- Fine-tuned model results.
- Improvement or regression.
- Safety evaluation results.
- Limitations.
- Recommendations.

Save the report under:

`artifacts/reports/`

## Model promotion gate

Implement three outcomes:

**PASS**

The model satisfies all configured mandatory quality checks.

**PASS WITH WARNINGS**

The model shows improvement but has issues requiring human review.

**FAIL**

The model fails mandatory quality or safety checks.

Make thresholds configurable.

Do not automatically publish a failed model.

PASS WITH WARNINGS must not automatically publish unless an explicitly configured and documented approval policy permits it.

A PASS is evidence of performance on the selected evaluation tests only. It is not evidence of clinical validation.

---

# 10. EXPERIMENT TRACKING

Use MLflow locally for experiment tracking.

Keep the setup simple.

Track:

- Base model name.
- Dataset identifier and version.
- Dataset fingerprint.
- Training parameters.
- LoRA configuration.
- Quantization configuration.
- Training loss.
- Validation loss.
- Evaluation metrics.
- Training duration.
- Git commit SHA.
- Model artifact location.
- GPU information.
- Library versions.

Log important training parameters and metrics from the training script.

Do not introduce paid experiment tracking services.

Provide simple instructions for starting the MLflow UI locally.

Use local MLflow storage initially.

Do not create a complicated tracking server infrastructure.

If MLflow adds unnecessary complexity to the first working implementation, implement the core logging first and integrate MLflow in a separate phase.

---

# 11. TRAINING MONITORING AND LOGGING

Monitoring should focus on the fine-tuning pipeline, not a web application.

Use simple, free, open-source tools.

## Python logging

Implement useful logging for:

- Dataset loading.
- Data preparation.
- Model loading.
- GPU detection.
- Quantization setup.
- Training start and completion.
- Training errors.
- Evaluation.
- Report generation.
- Model publishing.

Use appropriate log levels:

- DEBUG.
- INFO.
- WARNING.
- ERROR.

Avoid excessive logging.

Never log tokens, passwords or secret values.

## Training metrics

Track:

- Training loss.
- Validation loss.
- Learning rate.
- Training steps.
- Epoch progress.
- Training duration.
- Evaluation metrics.
- GPU information.

Use TensorBoard if it integrates cleanly with the Hugging Face Trainer.

MLflow should track experiment-level information.

TensorBoard can provide training metric visualization.

Do not introduce unnecessary monitoring platforms.

## GPU monitoring

Where supported, capture basic GPU information using tools such as `nvidia-smi`.

Do not build a complex GPU monitoring service.

Document how to inspect GPU utilization and memory during Kaggle training.

## Failure visibility

When training fails:

- Show the actual error.
- Save useful logs.
- Preserve available training history.
- Avoid marking the run successful.
- Explain the likely cause when possible.

Do not fabricate successful monitoring results.

---

# 12. DOCKER SUPPORT

I want Docker support if it genuinely improves reproducibility and development.

However, do not make Docker unnecessarily complicated.

## Docker requirements

Create a simple `Dockerfile` for:

- Local development.
- CPU-side dataset validation.
- Running unit tests.
- Running lightweight pipeline scripts.
- Reproducing the Python environment.

If a Docker Compose file provides a real benefit, create a simple `docker-compose.yml`.

One possible use is running the local MLflow tracking service.

Do not create unnecessary containers or microservices.

## Important GPU distinction

Actual GPU fine-tuning will run on Kaggle.

Do not assume that the Docker container will provide Kaggle GPU access.

Do not force Docker into the Kaggle training workflow.

Do not create a complicated CUDA Docker image unless there is a clear requirement.

Use the simplest reliable Docker setup for local validation and development.

## Docker security

- Do not copy `.env` or secrets into the image.
- Use `.dockerignore`.
- Use non-root execution where practical.
- Keep images reasonably small.
- Do not commit generated model weights into the Docker image.
- Use `.env.example` with placeholders only.

Document exact commands to build and run the container.

---

# 13. REUSE EXISTING KAGGLE + GITHUB ACTIONS CI/CD

This is a major requirement.

Preserve and extend my existing Kaggle + GitHub Actions architecture.

Do not replace it with a different training platform.

## GitHub Actions responsibilities

GitHub Actions should:

1. Check out the repository.
2. Set up Python.
3. Install CPU-compatible dependencies.
4. Run formatting and lint checks if configured.
5. Validate dataset structure using a small fixture or existing local data.
6. Run CPU-only pytest tests.
7. Stop if validation or tests fail.
8. Stage the required project code and configuration to the existing private Kaggle Dataset.
9. Use Kaggle CLI to queue the training notebook when manually triggered.
10. Report the Kaggle submission result.

GitHub Actions must not perform GPU fine-tuning.

Do not introduce unnecessary paid GitHub Actions services.

## Kaggle responsibilities

Kaggle should:

1. Receive the staged project code.
2. Load the configured dataset.
3. Detect GPU availability.
4. Run data preparation.
5. Run QLoRA fine-tuning.
6. Track training metrics.
7. Analyze training.
8. Evaluate the base model and fine-tuned model.
9. Run safety evaluation.
10. Generate reports.
11. Apply the promotion gate.
12. Publish the adapter only if publishing is enabled and the promotion policy permits it.

## Kaggle notebook

Create or update:

`kaggle/train.ipynb`

The notebook should orchestrate the Python scripts.

It should not contain a second implementation of the training logic.

The notebook should:

- Display GPU information.
- Install the required dependencies.
- Locate the staged project code.
- Run data preparation.
- Run training.
- Run training analysis.
- Run evaluation.
- Generate the report.
- Apply the promotion gate.
- Optionally publish the adapter.

Keep notebook cells short and understandable.

## Kaggle staging

Inspect and reuse the existing:

- `kaggle/` directory.
- `kaggle/stage_dataset.sh`.
- Kaggle notebook.
- Kaggle metadata.
- Kaggle dataset source configuration.
- Kaggle CLI commands.
- Existing launch and status scripts.

Do not silently replace existing configurations.

Ensure only the required project files are staged.

Do not stage:

- `.env`.
- Credentials.
- GitHub secrets.
- Unnecessary local artifacts.
- Large files that are already available through Kaggle datasets.

## Workflow triggers

Preserve path-based workflow triggers.

Documentation-only changes must not automatically consume GPU quota.

Relevant changes may include:

- Training scripts.
- Dataset preparation scripts.
- Evaluation scripts.
- Training configuration.
- Kaggle notebook.
- Kaggle staging scripts.
- Dependencies.
- GitHub workflow files.

Keep manual `workflow_dispatch` support.

Do not automatically start GPU training on every commit.

## Training completion tracking

There is an important limitation in the existing architecture:

A successful `kaggle kernels push` submission only confirms that the job was queued or accepted. It does not prove that training completed successfully.

Investigate the available Kaggle API capabilities before implementing completion tracking.

If reliable completion verification is practical:

- Check actual run status.
- Verify required output artifacts.
- Report actual training completion.

If reliable automated completion tracking is not practical:

- Clearly report that the job was queued.
- Provide the Kaggle run reference where available.
- Explain how to verify the run manually.
- Do not report training as successful merely because the submission succeeded.

---

# 14. HUGGING FACE HUB PUBLISHING

Make Hugging Face publishing optional.

Use the existing optional `HF_TOKEN` configuration.

Publish only if:

1. Training completes successfully.
2. Required evaluation completes.
3. Mandatory evaluation checks pass.
4. The promotion gate returns PASS.
5. Publishing is explicitly enabled.

Publish:

- LoRA adapter.
- Required model configuration.
- Model card.
- Base model information.
- Dataset provenance.
- Training configuration.
- Evaluation summary.
- Known limitations.
- Intended educational use.
- Medical safety disclaimer.

Do not publish the raw dataset or any data whose license does not permit redistribution.

Never print the Hugging Face token.

If publishing is disabled, save the adapter locally or as a Kaggle output artifact.

---

# 15. TESTING

Use pytest.

Keep CPU tests separate from GPU-dependent tests.

Create tests for:

- Dataset schema validation.
- Missing values.
- Duplicate detection.
- Data cleaning.
- Deterministic splitting.
- Train/test leakage.
- Configuration validation.
- Training configuration.
- Evaluation metrics.
- Promotion gate logic.
- Artifact generation.
- Publishing conditions.

Use small fixtures for tests.

Do not download a large model during normal CI tests.

Mock model loading and external services where appropriate.

GPU tests should be run separately on Kaggle when required.

GitHub Actions must not require a GPU.

Do not claim a test passed unless it actually ran successfully.

---

# 16. PROJECT STRUCTURE

Keep the project structure small and easy to understand.

Use this as a starting point, not a requirement to create unnecessary files:

medical-llm-finetuning/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── analyze_training.py
│   ├── evaluate.py
│   ├── report.py
│   └── upload_adapter.py
│
├── kaggle/
│   ├── train.ipynb
│   ├── kernel-metadata.json
│   ├── stage_dataset.sh
│   └── launch.py
│
├── tests/
│   ├── test_data.py
│   ├── test_training_config.py
│   └── test_evaluation.py
│
├── artifacts/
│   ├── reports/
│   ├── plots/
│   └── training/
│
├── mlruns/
│
├── .github/
│   └── workflows/
│       └── train.yml
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── .env.example
├── requirements.txt
├── requirements-ci.txt
└── README.md

Adjust the structure based on the existing repository.

Do not create every file simply because it appears in this example.

For example:

- Keep `docker-compose.yml` only if it is useful.
- Keep `upload_adapter.py` only if publishing is implemented.
- Avoid creating separate files for trivial helper functions.
- Reuse existing files where possible.

---

# 17. CONFIGURATION AND DEPENDENCIES

Use one main YAML configuration file for training parameters.

Example:

configs/config.yaml

Keep it beginner-friendly.

Use separate dependency files only where they serve a clear purpose:

- `requirements.txt` for the training environment.
- `requirements-ci.txt` for lightweight CPU validation.

Pin compatible versions after testing them in the actual environment.

Avoid unnecessary dependencies.

Document why important libraries are required.

Do not install GPU-only dependencies into CPU CI unnecessarily.

Do not blindly copy dependency versions from another project without checking compatibility.

---

# 18. FREE-OF-COST REQUIREMENT

I want to build this project with zero or minimal cost.

Prefer:

- Kaggle free GPU resources.
- GitHub Actions free allowances where available.
- Open-source Python libraries.
- Hugging Face open-weight models with suitable licenses.
- Local MLflow.
- TensorBoard.
- Docker.
- Local development tools.

Do not introduce paid APIs or paid infrastructure.

Do not assume that free services provide unlimited resources.

Document relevant limitations, quotas and hardware constraints.

If a feature requires a paid service, stop and explain the limitation and suggest a free alternative.

---

# 19. DOCUMENTATION

Create a clear README.md.

It should explain:

1. Project objective.
2. Scope and limitations.
3. Architecture.
4. Dataset source.
5. Dataset license and provenance.
6. Base model selection.
7. Project structure.
8. Python environment setup.
9. Dependency installation.
10. Dataset preparation.
11. Local CPU validation.
12. Docker setup.
13. Kaggle configuration.
14. GitHub Actions configuration.
15. Training execution.
16. Training monitoring.
17. Evaluation.
18. Promotion gate.
19. Hugging Face publishing.
20. Common errors and troubleshooting.

Include actual commands.

Do not document functionality that has not been implemented.

## Learning documentation

I am learning LLM engineering, so explain the important concepts in beginner-friendly language.

Include explanations of:

- Tokens and tokenization.
- Next-token prediction.
- Supervised fine-tuning.
- Loss.
- Training and validation loss.
- Attention.
- LoRA.
- QLoRA.
- Low-rank adaptation.
- 4-bit quantization.
- NF4.
- Gradient accumulation.
- Gradient checkpointing.
- Mixed precision.
- Epochs.
- Batch size.
- Learning rate.
- Overfitting.
- Model evaluation.
- Adapter weights.
- Experiment tracking.

Explain where these concepts appear in the actual project code.

Do not create a huge theoretical textbook. Keep explanations relevant to the implementation.

---

# 20. ARCHITECTURE

The final architecture should remain focused on fine-tuning.

Use this architecture as the intended design:

Developer
    |
    v
GitHub Repository
    |
    v
GitHub Actions
    |
    +--> Code validation
    |
    +--> Dataset validation
    |
    +--> CPU tests
    |
    +--> Manual training trigger
              |
              v
          Kaggle CLI
              |
              v
       Private Kaggle Dataset
              |
              v
        Kaggle GPU Notebook
              |
              v
       Dataset Preparation
              |
              v
       QLoRA Fine-tuning
              |
              +--> Python logging
              |
              +--> TensorBoard
              |
              +--> MLflow
              |
              v
       Training Analysis
              |
              v
       Model Evaluation
              |
              v
       Evaluation Report
              |
              v
       Promotion Gate
          /       
       PASS       FAIL
        |           |
        v           v
  Save Adapter   No Publishing
        |
        v
 Optional Hugging Face Hub

Docker:
Local development, CPU validation and testing.

Docker must not be a required part of Kaggle GPU training.

Create an updated Mermaid architecture diagram in the README.

---

# 21. IMPLEMENTATION PHASES

Work in small phases.

Do not implement the entire project in one large operation.

After each phase, explain what changed and wait for my approval before continuing.

## Phase 0: Existing workspace inspection

- Inspect the repository.
- Read the README.
- Inspect Python scripts.
- Inspect Kaggle integration.
- Inspect GitHub workflows.
- Inspect dataset staging.
- Inspect dependencies.
- Inspect Docker and monitoring setup if present.
- Identify what can be reused.
- Identify what needs modification.

Do not modify files during Phase 0.

Provide:

1. Current architecture.
2. Existing reusable components.
3. Missing components.
4. Problems or risks.
5. Proposed simple project structure.
6. Proposed implementation plan.
7. Any questions that genuinely require my input.

STOP and wait for my approval.

## Phase 1: Dataset preparation

After approval:

- Verify MedQuAD provenance and license.
- Download and inspect the dataset.
- Implement data validation.
- Implement cleaning and duplicate detection.
- Implement leakage-safe train/validation/test splitting.
- Generate processed JSONL files.
- Generate dataset statistics.
- Add CPU tests.

Run the tests and show actual results.

STOP and wait for approval.

## Phase 2: Model selection and QLoRA training

After approval:

- Finalize the base model.
- Verify license and hardware compatibility.
- Implement model loading.
- Implement QLoRA configuration.
- Implement supervised fine-tuning.
- Add GPU detection.
- Add configurable training parameters.
- Save the adapter and training history.
- Add training logging.

STOP and wait for approval.

## Phase 3: Training analysis and evaluation

After approval:

- Implement loss curve generation.
- Implement training analysis.
- Implement base-vs-fine-tuned evaluation.
- Add medical safety evaluation cases.
- Implement the promotion gate.
- Generate reports.

STOP and wait for approval.

## Phase 4: Experiment tracking and monitoring

After approval:

- Integrate MLflow.
- Add TensorBoard if useful.
- Improve training logs.
- Track model and dataset metadata.
- Track hardware and training duration.
- Document how to inspect the results locally.

Keep this phase lightweight.

STOP and wait for approval.

## Phase 5: Docker support

After approval:

- Create a simple Dockerfile.
- Add `.dockerignore`.
- Add Docker Compose only if it provides a genuine benefit.
- Test CPU validation inside Docker.
- Optionally configure local MLflow tracking through Compose.
- Document commands.

Do not attempt to use Docker as a replacement for Kaggle GPU training.

STOP and wait for approval.

## Phase 6: Kaggle and GitHub Actions CI/CD

After approval:

- Reuse the existing Kaggle staging architecture.
- Update the notebook to orchestrate Python scripts.
- Update Kaggle metadata only where required.
- Update staging scripts only where required.
- Update GitHub Actions.
- Preserve CPU-only CI validation.
- Keep manual GPU training triggers.
- Preserve path-based triggers.
- Add accurate Kaggle queue/completion reporting.
- Test workflow logic without consuming GPU quota where possible.

STOP and wait for approval.

## Phase 7: Optional model publishing and final documentation

After approval:

- Implement promotion-gated Hugging Face adapter publishing.
- Verify secrets handling.
- Update the README.
- Update architecture documentation.
- Document the complete execution workflow.
- Document limitations and troubleshooting.
- Perform a final repository review.

Do not deploy or publish publicly without my approval.

STOP after Phase 7.

---

# 22. IMPORTANT CODEX RULES

Before every phase:

1. Inspect the current implementation.
2. Understand existing code before changing it.
3. Reuse working components.
4. Identify dependencies.
5. Explain the implementation plan.
6. Identify any potential breaking changes.
7. Keep the implementation as simple as possible.

After every phase:

1. List files created.
2. List files modified.
3. Explain the code and important decisions.
4. Explain the relevant LLM/ML concepts in beginner-friendly language.
5. Provide exact commands to run.
6. Run relevant tests.
7. Show actual test results.
8. Explain failures.
9. Document incomplete tasks.
10. Mention any cost, quota or hardware limitations.

Additional rules:

- Never print credentials.
- Never hardcode secrets.
- Never silently overwrite existing working files.
- Never delete existing CI/CD functionality without explaining why and receiving approval.
- Never fabricate training results.
- Never fabricate evaluation scores.
- Never claim training succeeded just because Kaggle accepted a job.
- Never claim a model is medically validated.
- Never publish a failed model.
- Never assume that the dataset or model license permits unrestricted use.
- Never add paid infrastructure without approval.
- Never overcomplicate the codebase.
- Do not move to the next phase automatically.
- Ask only essential clarification questions.

If a feature is unnecessary for the current fine-tuning pipeline, do not implement it just because it is technically possible.

---

# FINAL EXPECTATION

I want to complete this project with practical knowledge and hands-on experience in:

- LLM fundamentals.
- Medical dataset engineering.
- Tokenization.
- Supervised Fine-Tuning.
- PyTorch.
- Hugging Face Transformers.
- PEFT.
- LoRA.
- QLoRA.
- 4-bit quantization.
- GPU training.
- Model evaluation.
- Training analysis.
- Experiment tracking.
- MLflow.
- TensorBoard.
- Python logging.
- Docker.
- GitHub Actions.
- Kaggle GPU.
- Model versioning.
- Model promotion.
- Hugging Face Hub.
- MLOps fundamentals.

The final project should be:

- Simple and beginner-friendly.
- Production-style in engineering quality.
- Reproducible.
- Testable.
- Maintainable.
- Free or minimal-cost.
- Suitable for a practical AI/ML engineering portfolio.

I prefer a small, well-written project that I can explain confidently in an interview over a huge, complicated codebase that I do not understand.

## START NOW

**Start with Phase 0 only.**

Inspect my current workspace and existing Kaggle + GitHub CI/CD implementation.


Show me your findings and proposed changes.

