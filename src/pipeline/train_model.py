"""
Code followed by a # was edited by ChatGPT 5.1 at 5:21 PM on 11/23/25 to make this file compliant with Weights 
and Biases Deployment rather than MLFlow (original lines of code were written for MLFlow without AI, but modified 
later by ChatGPT 5.1 to work with Weights and Biases instead).
"""

import yaml
import pandas as pd
import json
import joblib
import os
import wandb   #

from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from src.utils.logger import get_logger

def load_config():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def update_config_with_best_params(cfg, best_params):
    model_type = cfg["model"]["type"]
    cfg["model"]["params"][model_type] = best_params
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def get_model(model_type, params):
    if model_type == "random_forest":
        return RandomForestClassifier(**params)
    elif model_type == "xgboost":
        return XGBClassifier(**params)
    else:
        raise ValueError("Unknown model type.")


def train_model():
    cfg = load_config()
    project_root = Path(__file__).resolve().parents[2]

    logs_dir = project_root / cfg["logging"]["logs"]
    checkpoints_dir = project_root / cfg["logging"]["checkpoints"]
    trained_dir = project_root / cfg["logging"]["trained"]
    results_dir = project_root / cfg["logging"]["results"]

    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    trained_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("training", str(logs_dir / "training.log"))

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:  #
        wandb.login(key=wandb_key)  #

    wandb.init(project="dementia-ml", name="training")   #

    data_path = project_root / cfg["data"]["path"]
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Group"])
    y = df["Group"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=cfg["data"]["test_ratio"] + cfg["data"]["val_ratio"],
        stratify=y,
        random_state=cfg["data"]["random_seed"]
    )

    val_ratio_adj = cfg["data"]["val_ratio"] / (cfg["data"]["test_ratio"] + cfg["data"]["val_ratio"])

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_ratio_adj,
        stratify=y_temp,
        random_state=cfg["data"]["random_seed"]
    )

    model_type = cfg["model"]["type"]
    params = cfg["model"]["params"][model_type]

    wandb.log({"params": params})   #

    model = get_model(model_type, params)

    if cfg["tuning"]["enabled"]:
        search_space = {
            "n_estimators": [100, 200, 400, 600],
            "max_depth": [3, 4, 5, 6, 8],
        }

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=search_space,
            n_iter=cfg["tuning"]["n_iter"],
            cv=3,
            n_jobs=-1,
            verbose=2
        )

        search.fit(X_train, y_train)

        best_params = search.best_params_
        model = search.best_estimator_

        wandb.log({"best_params": best_params})   #
        update_config_with_best_params(cfg, best_params)

    model.fit(X_train, y_train)

    checkpoint_path = checkpoints_dir / f"{model_type}_checkpoint.pkl"
    joblib.dump(model, checkpoint_path)

    artifact = wandb.Artifact("checkpoint", type="model")   #
    artifact.add_file(str(checkpoint_path))   #
    wandb.log_artifact(artifact)   #

    val_acc = accuracy_score(y_val, model.predict(X_val))
    wandb.log({"val_accuracy": val_acc})   #

    final_path = trained_dir / f"{model_type}_final.pkl"
    joblib.dump(model, final_path)

    final_art = wandb.Artifact("final_model", type="model")   #
    final_art.add_file(str(final_path))   #
    wandb.log_artifact(final_art)   #

    api_model_base_path = project_root / "app" 
    api_model_path = api_model_base_path / "model.pkl"
    api_model_base_path.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, api_model_path)

    results = {"val_accuracy": val_acc}
    results_file = results_dir / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    results_art = wandb.Artifact("training_results", type="results")   #
    results_art.add_file(str(results_file))   #
    wandb.log_artifact(results_art)   #


if __name__ == "__main__":
    train_model()
