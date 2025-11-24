# Dementia Classification MLOps Project

This repository implements an end to end MLOps workflow for a dementia classification model, including:

- Data ingestion, cleaning, and feature engineering
- Model training and evaluation with logging
- Packaged model artifact
- FastAPI inference service
- Docker containerization
- Cloud deployment with a front end on Hugging Face Spaces

The model predicts a binary dementia label from 14 normalized clinical and imaging features derived from the OASIS style dementia dataset.

---

## Project Structure

```
.
├── README.md
├── app/
│   ├── app.py              # FastAPI inference API (local)
│   ├── model.pkl           # Trained XGBoost model used by the API
│   └── space_app.py        # Gradio + FastAPI app used on Hugging Face Space
├── config/
│   └── config.yaml         # Central configuration for paths and logging
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   │   ├── cleaned_dementia_dataset.csv
│   │   ├── engineered_dementia_dataset.csv
│   │   └── preprocessed_dementia_dataset.csv
│   └── raw/
│       └── dementia_dataset.csv
├── dockerfile              # Dockerfile for containerizing the local FastAPI app
├── docs/
│   └── data_plots/         # Exploratory data analysis figures
├── models/
│   ├── checkpoints/
│   │   └── xgboost_checkpoint.pkl
│   └── trained/
│       └── xgboost_final.pkl
├── outputs/
│   ├── logs/
│   │   ├── evaluation.log
│   │   └── training.log
│   └── results/
│       ├── classification_report.json
│       ├── cm.png
│       ├── evaluation_results.json
│       └── training_results.json
├── requirements.txt        # Full project dependencies (training, EDA, etc.)
├── requirements-api.txt    # Minimal dependencies for the API image
└── src/
    ├── data/               # Data analysis and cleaning scripts
    ├── models/             # Training and evaluation scripts
    ├── pipeline/           # End to end data pipeline
    └── utils/              # Logging and plotting utilities
```

## Environment Setup

Create and activate a virtual environment, then install dependencies.

```
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```
## Datasets
The dataset (under `data/raw/dementia_dataset.csv`) contains clinical and structural MRI derived features for patients with and without dementia.

Key variables used by the model (after preprocessing and scaling):

- Visit
- MR Delay
- M/F
- Age
- EDUC
- SES
- MMSE
- CDR
- eTIV
- nWBV
- ASF
- ABV
- CII
- CDR_RATE

The target label is a binary dementia indicator (for example Group in the original dataset), which is not passed to the model at inference time.

Preprocessed versions of the dataset are saved under `data/processed/`:

- preprocessed_dementia_dataset.csv
- cleaned_dementia_dataset.csv
- engineered_dementia_dataset.csv


# Environment Setup

Create and activate a virtual environment, then install dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

To install only minimal API dependencies:

```
pip install -r requirements-api.txt
```


# Configuration

Global configuration lives in `config/config.yaml`. It defines:

- Data directories
- Model artifact directories
- Logging output paths
- Results directory

Training and evaluation scripts read this configuration.


# Data Pipeline

Pipeline scripts are in `src/pipeline` and `src/data`.

Flow:

1. Ingest:
   ```
   python src/pipeline/ingest_data.py
   ```
2. Preprocess:
   ```
   python src/pipeline/preprocess_data.py
   ```
3. Feature engineer:
   ```
   python src/pipeline/feature_engineer_data.py
   ```
4. Or run the orchestrated pipeline:
   ```
   python src/pipeline/pipeline.py
   ```


# Model Training and Evaluation

Training (`src/models/train_model.py`) saves:

- `models/checkpoints/xgboost_checkpoint.pkl`
- `models/trained/xgboost_final.pkl`

Evaluation (`src/models/evaluate_model.py`) produces:

- `outputs/results/classification_report.json`
- `outputs/results/evaluation_results.json`
- `outputs/results/cm.png`
- logs under `outputs/logs/`

Example:

```
python src/models/train_model.py
python src/models/evaluate_model.py
```


# Local FastAPI Inference

`app/app.py` loads the trained model and exposes:

- GET `/health`
- POST `/predict` returning:
  - predicted class
  - probability

Run locally:

```
uvicorn app.app:app --reload
```

Local endpoints:

```
http://localhost:8000/health
http://localhost:8000/predict
```

Example curl:

```
curl -X POST "http://localhost:8000/predict"   -H "Content-Type: application/json"   -d '{ "Visit": 0.0, "MR_Delay": 0.0, "M_F": 1, "Age": 0.736842105263158,
        "EDUC": 0.11764705882352944, "SES": 0.75, "MMSE": 0.8076923076923077,
        "CDR": 0.25, "eTIV": 0.6069042316258353, "nWBV": 0.08290155440414493,
        "ASF": 0.26300984528832627, "ABV": 0.05031330417623496,
        "CII": -0.5576923076923077, "CDR_RATE": 0.0 }'
```


# Docker Deployment

Dockerfile:

```
FROM python:3.11-slim
WORKDIR /app
COPY requirements-api.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
ENV PORT=8080
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

Build and run:

```
docker build -t dementia-api .
docker run -p 8080:8080 dementia-api
```


# Cloud Deployment (Hugging Face Spaces)

Space URL:

```
https://dtanzillo-dementia-ml.hf.space/
```

API:

```
https://dtanzillo-dementia-ml.hf.space/api/health
https://dtanzillo-dementia-ml.hf.space/api/predict
```

Python example:

```python
import requests
url = "https://dtanzillo-dementia-ml.hf.space/api/predict"
payload = { "Visit": 0.0, "MR_Delay": 0.0, "M_F": 1, "Age": 0.736842105263158,
            "EDUC": 0.11764705882352944, "SES": 0.75, "MMSE": 0.8076923076923077,
            "CDR": 0.25, "eTIV": 0.6069042316258353, "nWBV": 0.08290155440414493,
            "ASF": 0.26300984528832627, "ABV": 0.05031330417623496,
            "CII": -0.5576923076923077, "CDR_RATE": 0.0 }
print(requests.post(url, json=payload).text)
```


# Gradio Front End

`app/space_app.py` provides:

- Slider and dropdown UI for all features
- Final class prediction
- Probability output


# Reproducibility Summary

- Paths and constants from `config/config.yaml`
- All dependencies pinned
- Model artifact reused across local, Docker, and cloud deployments
- Complete pipeline from raw data to deployed inference preserved