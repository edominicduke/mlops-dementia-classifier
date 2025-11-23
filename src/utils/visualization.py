import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, labels, save_path=None):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
