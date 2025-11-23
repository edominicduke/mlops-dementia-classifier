import yaml
import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def update_config_with_best_params(cfg, best_params):
    model_type = cfg["model"]["type"]
    cfg["model"]["params"][model_type] = best_params

    # Save back into config/config.yaml
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


def main():
    cfg = load_config()
    logger = get_logger("training", cfg["logging"]["logs"] + "training.log")

    logger.info("Loading dataset...")
    df = pd.read_csv(cfg["data"]["path"])

    X = df.drop(columns=["Group"])
    y = df["Group"]

    # train to validation to test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=cfg["data"]["test_ratio"] + cfg["data"]["val_ratio"],
        stratify=y,
        random_state=cfg["data"]["random_seed"]
    )

    val_ratio_adj = cfg["data"]["val_ratio"] / (
        cfg["data"]["test_ratio"] + cfg["data"]["val_ratio"]
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_ratio_adj,
        stratify=y_temp,
        random_state=cfg["data"]["random_seed"]
    )

    logger.info("Data split complete.")

    # Model setup
    model_type = cfg["model"]["type"]
    params = cfg["model"]["params"][model_type]

    model = get_model(model_type, params)
    logger.info(f"Model initialized: {model_type}")

    # Hyperparameter search
    if cfg["tuning"]["enabled"]:
        logger.info("Starting hyperparameter tuning...")

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

        logger.info(f"Best params found: {best_params}")

        # Update YAML with optimal hyperparameters
        update_config_with_best_params(cfg, best_params)
        logger.info("config.yaml updated with optimized hyperparameters.")

    # Train final model
    logger.info("Training final model...")
    model.fit(X_train, y_train)

    # Ensure directories exist
    Path(cfg["logging"]["checkpoints"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["logging"]["trained"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["logging"]["logs"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["logging"]["results"]).mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    checkpoint_path = Path(cfg["logging"]["checkpoints"]) / f"{model_type}_checkpoint.pkl"
    joblib.dump(model, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

    # Evaluate on validation
    val_acc = accuracy_score(y_val, model.predict(X_val))
    logger.info(f"Validation Accuracy: {val_acc:.4f}")

    # Save final model
    final_path = Path(cfg["logging"]["trained"]) / f"{model_type}_final.pkl"
    joblib.dump(model, final_path)
    logger.info(f"Final model saved to {final_path}")

    # Save metrics
    results = {"val_accuracy": val_acc}

    with open(cfg["logging"]["results"] + "training_results.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
