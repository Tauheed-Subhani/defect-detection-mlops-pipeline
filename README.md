# Industrial Defect Detection — MLOps Pipeline

An end-to-end MLOps pipeline for binary image classification, built and demonstrated on an industrial casting defect dataset. The pipeline covers model training, experiment tracking, API serving, containerization, automated drift detection, and CI/CD — with a persistence-based retraining trigger and a documented, honest account of what is automated versus what remains a manual, human-reviewed step.

**Demonstrated on:** casting product defect classification (defective vs. ok)
**Architecture is dataset-agnostic** — see [Dataset-Agnostic Design](#dataset-agnostic-design) below.

---

## What this project actually does

| Stage | What it does | Status |
|---|---|---|
| Training | Fine-tunes a ResNet18 (ImageNet-pretrained) for binary classification | ✅ Working — 99.58% test accuracy |
| Serving | FastAPI `/predict` endpoint returns label + confidence score | ✅ Working |
| Experiment tracking | MLflow logs training params/metrics (SQLite backend) | ✅ Working |
| Containerization | Dockerfile packages the FastAPI app for portable deployment | ✅ Working |
| Drift detection | Evidently-based comparison of training vs. incoming data distributions | ✅ Working, automated |
| Retrain trigger | Persistence-based rule (2 consecutive drifted runs) decides if retraining is warranted | ✅ Working, automated |
| Retraining | Fine-tunes existing model on new + replayed old data (Colab, GPU) | ⚠️ Prototyped manually, not integrated into the automated pipeline |
| Model promotion (Champion/Challenger) | Compares new vs. current model before promoting | ⚠️ Prototyped, known evaluation issues found (see [Known Limitations](#known-limitations)) — not automated |
| CI/CD | GitHub Actions: installs dependencies, checks code compiles on every push | ✅ Working |

---

## Why this scope, and what "automated" actually means here

MLOps does not mean every step runs without a human — it means applying engineering discipline (tracking, monitoring, reproducibility, CI) to the ML lifecycle, with automation applied *where it's safe and well-validated* to apply it.

In this project:
- **Genuinely automated, with no human judgment required at runtime:** drift detection, the persistence-based retrain trigger, MLflow logging, CI checks, and containerized serving.
- **Deliberately kept manual:** the actual retraining run and the decision to promote a new model. This was a conscious choice, not an oversight — see below.

---

## Known Limitations (and why they're documented, not hidden)

While prototyping the Champion/Challenger model comparison, two issues were identified:

1. **Evaluation leakage:** the "challenger" model was fine-tuned on a perturbed version of the same images used in the original test set (used to simulate production drift). This gives the challenger an indirect memorization advantage when evaluated on the original test set, making a naive accuracy comparison misleading.
2. **Possible overfitting:** the fine-tuned model showed suspiciously high accuracy (>99%) and perfect recall on both the original and drifted test data — a pattern more consistent with memorization than genuine generalization improvement, given the small fine-tuning dataset (~1,465 images).

Rather than ship an automated promotion decision built on an evaluation setup with these issues, retraining and promotion were kept as manual, human-reviewed steps. In a production setting, this would be addressed with a properly isolated holdout set that never contributes — directly or indirectly — to any training or fine-tuning data.

---

## Dataset-Agnostic Design

The pipeline contains no casting-specific logic anywhere in the code:

- **Training (`train_local.py`)** uses `torchvision.ImageFolder`, which only requires two class-subfolders of images (e.g. `def_front/`, `ok_front/`) — any binary image classification dataset with this folder structure works without code changes.
- **Model** is a standard ResNet18 with a 2-class output head — no casting-specific architecture choices.
- **Serving, tracking, containerization, and drift detection** all operate on generic image statistics and predictions — nothing is hardcoded to defect appearance.

**To point this pipeline at a new dataset:**
- Binary classification: change the data folder path(s) — one config change.
- Multi-class classification: change the data folder path(s) **and** the output layer size (`nn.Linear(model.fc.in_features, N)`) — two small changes.

---

## Project Structure

```
PipeLine/
├── archive/                          # Dataset (not committed — see Setup)
│   └── casting_data/casting_data/
│       ├── train/
│       │   ├── def_front/
│       │   └── ok_front/
│       └── test/
│           ├── def_front/
│           └── ok_front/
├── production_stream/                # Simulated drifted data (generated, not committed)
├── .github/workflows/ci.yml          # CI pipeline
├── train_local.py                    # Training + MLflow logging
├── main.py                           # FastAPI serving
├── simulate_drift.py                 # Generates perturbed "production" data
├── drift_monitor.py                  # Evidently-based drift detection
├── retrain_trigger.py                # Persistence-based retrain decision logic
├── Dockerfile
├── requirements.txt
├── model.pth                         # Trained weights (not committed — see Setup)
├── mlflow.db                         # MLflow tracking store (not committed, regenerated locally)
└── .gitignore
```

---

## Setup — Reproducing This Project

This repo intentionally does not include the dataset or trained model weights (too large for version control). To reproduce:

1. **Clone the repo:**
   ```
   git clone https://github.com/Tauheed-Subhani/defect-detection-mlops-pipeline.git
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Get the dataset:** download the [Casting Product Defect Dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) from Kaggle, and place it at `archive/casting_data/casting_data/` matching the structure shown above.

4. **Train the model:**
   ```
   python train_local.py
   ```
   This produces `model.pth` and logs the run to `mlflow.db`.

5. **Serve predictions:**
   ```
   uvicorn main:app --reload
   ```
   Visit `http://127.0.0.1:8000/docs` for the interactive API.

6. **View experiment tracking:**
   ```
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

7. **Simulate drift and check detection:**
   ```
   python simulate_drift.py
   python drift_monitor.py
   python retrain_trigger.py
   ```

8. **Run via Docker (optional):**
   ```
   docker build -t defect-detection-api .
   docker run -p 8000:8000 defect-detection-api
   ```

---

## CI/CD

Every push to `main` triggers a GitHub Actions workflow (`.github/workflows/ci.yml`) that:
- Installs dependencies from `requirements.txt`
- Verifies all Python files compile without syntax errors

**Note:** since the dataset and trained model are intentionally excluded from version control, CI validates code health (dependencies, syntax) rather than full pipeline execution. Running the full pipeline requires the local setup steps above.

---

## Future Work

- Isolate a true holdout set (never touched by any perturbation or fine-tuning step) to enable a trustworthy, automated Champion/Challenger promotion decision
- Wire the retrain trigger directly into an automated retraining job (currently a manual Colab step)
- Confidence-based routing for low-confidence predictions to a human review queue, as a complement to batch-level drift detection
- Cloud deployment (e.g. Render/Railway) with `model.pth` hosted via GitHub Releases or cloud storage, fetched at container startup
- Periodic random-sample auditing of high-confidence predictions, to catch silent failure modes drift detection alone might miss

---

## Tech Stack

Python · PyTorch · torchvision · FastAPI · MLflow · Docker · Evidently AI · GitHub Actions
