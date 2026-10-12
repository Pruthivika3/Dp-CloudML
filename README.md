# DP-CloudML — Differential Privacy ML Dashboard

A dashboard-first educational platform for measuring the privacy–utility tradeoff of Differentially Private SGD on **user-uploaded tabular datasets**.

## What changed

- Upload **CSV / XLSX / XLS** directly from the dashboard.
- Select the target/label column in the UI.
- Automatic preprocessing for numeric and categorical features.
- Automatic train/test split and standardization.
- Adaptive tabular MLP whose input/output dimensions match the uploaded dataset.
- Baseline vs DP-SGD comparison using the same prepared dataset.
- Faster defaults: 3 epochs, batch size 256, Adam optimizer, and a smaller DP experiment set.
- DP epsilon is calculated with the Opacus privacy accountant.
- Dataset rows are capped at 50,000 and preprocessed features at 512 to keep dashboard experiments practical.

## Supported data

The current pipeline is for **classification**. The selected target must contain 2–100 classes. Numeric and categorical feature columns are supported. Columns that are almost entirely unique strings are treated as identifier-like columns and removed automatically.

Do not upload sensitive personal data unless your deployment and data-governance process explicitly permits it.

## Run locally — Windows

```powershell
cd DP-CloudML
python -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
python backend\app.py
```

Open `http://127.0.0.1:5000`.

### Dashboard workflow

1. Choose a CSV/XLSX/XLS file.
2. Click **Upload & Inspect**.
3. Select the target/label column.
4. Choose Baseline or Differential Privacy.
5. Train one configuration or run the full experiment.
6. Review accuracy, F1, epsilon, training time and memory.

## API

```text
GET  /api/health
GET  /api/config
GET  /api/results
GET  /api/metrics
POST /api/upload
POST /api/train
POST /api/run-experiment
POST /api/reset
```

`/api/upload` accepts multipart form field `file` and returns a dataset ID, columns and a small preview. Training requests then use that dataset ID plus `label_column`.

## Experiment configurations

The full experiment runs:

- Baseline
- DP noise multiplier 0.5
- DP noise multiplier 1.0
- DP noise multiplier 1.5

The exact epsilon and utility values are generated at runtime. Do not put invented measurements in a report.

## Docker

```powershell
docker build -t dp-cloudml:latest .
docker run --rm -p 5000:5000 dp-cloudml:latest
```

Or:

```powershell
docker compose up --build
```

For persistent uploaded datasets/results during local Docker use, mount `./data` and `./results` as volumes.

## Kubernetes

The supplied manifests still expose the Flask service and support replicas/HPA. For production use with multiple replicas, move uploaded datasets and results to shared object/database storage rather than relying on each pod's local filesystem.

```powershell
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

## Privacy interpretation

Gradient clipping limits the contribution of an individual example and calibrated noise is added through DP-SGD. Epsilon is a privacy-loss parameter under the accountant and training assumptions. Smaller epsilon generally represents a stronger privacy guarantee when the other assumptions, including delta, are held fixed.

Kubernetes provides orchestration, replicas and scaling; it does **not** provide the differential privacy guarantee.

This project is an educational prototype. A research or production deployment should document the exact sampling, accountant, clipping, noise, delta, number of steps and data-governance assumptions.
