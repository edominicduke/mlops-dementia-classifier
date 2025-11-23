import json
import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.utils.logger import get_logger
from src.utils.visualization import plot_confusion_matrix

def load_config():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    project_root = Path(__file__).resolve().parents[2]

    # Directories (absolute paths)
    logs_dir = project_root / cfg["logging"]["logs"]
    results_dir = project_root / cfg["logging"]["results"]
    trained_dir = project_root / cfg["logging"]["trained"]

    # Create directories if they do not exist
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    trained_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    log_file = logs_dir / "evaluation.log"
    logger = get_logger("evaluation", str(log_file))

    logger.info("Loading dataset...")

    # Load dataset
    data_path = project_root / cfg["data"]["path"]
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Group"])
    y = df["Group"]

    # Load trained model
    model_path = trained_dir / f"{cfg['model']['type']}_final.pkl"
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    # Predictions
    preds = model.predict(X)

    # Metrics
    acc = accuracy_score(y, preds)
    logger.info(f"Full-dataset accuracy: {acc:.4f}")

    report = classification_report(y, preds, output_dict=True)

    # Save evaluation results
    results_file = results_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=4)
    logger.info(f"Saved evaluation results to {results_file}")

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    cm_path = results_dir / "cm.png"
    plot_confusion_matrix(cm, ["No Dementia", "Dementia"], save_path=str(cm_path))
    logger.info(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
