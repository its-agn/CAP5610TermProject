"""Shared evaluation, plotting, and results logging."""

import os
from datetime import datetime

from .data import LABEL_NAMES

def compute_metrics(y_true, y_pred):
    """Compute standard multi-class classification metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro"),
        "macro_recall": recall_score(y_true, y_pred, average="macro"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }

def print_metrics(metrics, model_name, y_true=None, y_pred=None):
    """Print formatted evaluation results. Includes classification_report if y_true/y_pred given."""
    from sklearn.metrics import classification_report

    print("=" * 60)
    print(f"{model_name} - Evaluation Results")
    print("=" * 60)
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score:  {metrics['macro_f1']:.4f}")
    print()
    if y_true is not None and y_pred is not None:
        print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))

def plot_confusion_matrix(y_true, y_pred, save_path, model_name):
    """Save a confusion matrix heatmap to disk."""
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name} - {date}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def save_results(model_name, metrics, elapsed, log_path, final=False):
    """Update a results_log.md with the latest run for this model.

    Each model/mode combo (e.g. "Random Forest (final)") gets its own section.
    Running again overwrites that section with new results.
    """
    section = f"{model_name} ({'final' if final else 'validation'})"
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_lines = [
        "",
        f"- **Date:** {date}",
        f"- **Accuracy:** {metrics['accuracy']:.4f}",
        f"- **Macro Precision:** {metrics['macro_precision']:.4f}",
        f"- **Macro Recall:** {metrics['macro_recall']:.4f}",
        f"- **Macro F1:** {metrics['macro_f1']:.4f}",
        f"- **Time:** {elapsed:.1f}s ({elapsed/60:.1f}m)",
    ]

    existing: dict[str, list[str]] = {}
    if os.path.exists(log_path):
        with open(log_path) as f:
            current_key = None
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("## "):
                    current_key = line[3:]
                    existing[current_key] = []
                elif current_key is not None:
                    existing[current_key].append(line)

    for key in existing:
        while existing[key] and existing[key][-1] == "":
            existing[key].pop()

    existing[section] = new_lines

    with open(log_path, "w") as f:
        f.write("# Results Log\n")
        for key, lines in existing.items():
            f.write(f"\n## {key}\n")
            for line in lines:
                f.write(f"{line}\n")

    print(f"Results saved to {log_path}")
