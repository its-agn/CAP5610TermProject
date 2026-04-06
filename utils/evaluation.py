"""Shared evaluation, plotting, and results logging."""

import os
import platform
from datetime import datetime

from .data import LABEL_NAMES

def _get_cpu_name():
    """Get the CPU model name."""
    # Linux / WSL
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except FileNotFoundError:
        pass
    # macOS
    import subprocess
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Windows — use wmic to get CPU name
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
                if len(lines) >= 2:
                    return lines[1]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return platform.processor() or "Unknown CPU"

def get_device_name(cpu_only=False):
    """Detect the compute device (GPU if available, otherwise CPU name).
    Pass cpu_only=True for models that only run on CPU (e.g. sklearn).
    """
    if not cpu_only:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "Apple MPS"
        except ImportError:
            pass
    return _get_cpu_name()

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

def save_results(model_name, metrics, elapsed, log_path, final=False, device=None,
                 default_params=False, params=None):
    """Update a results_log.md with the latest run for this model.

    Each model/mode combo (e.g. "Random Forest (final)") gets its own section.
    Running again overwrites that section with new results.
    Pass default_params=None to omit the params tag (e.g. for Decision Tree).
    """
    mode = "final" if final else "validation"
    if default_params is None:
        section = f"{model_name} ({mode})"
    else:
        params_tag = "default params" if default_params else "best params"
        section = f"{model_name} ({mode}, {params_tag})"
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    if device is None:
        device = get_device_name()
    new_lines = [
        "",
        f"- **Date:** {date}",
        f"- **Device:** {device}",
        f"- **Accuracy:** {metrics['accuracy']:.4f}",
        f"- **Macro Precision:** {metrics['macro_precision']:.4f}",
        f"- **Macro Recall:** {metrics['macro_recall']:.4f}",
        f"- **Macro F1:** {metrics['macro_f1']:.4f}",
        f"- **Time:** {elapsed:.1f}s ({elapsed/60:.1f}m)",
    ]
    if params:
        new_lines.append("")
        new_lines.append("<details>")
        new_lines.append("<summary>Parameters</summary>")
        new_lines.append("")
        for k, v in params.items():
            new_lines.append(f"- `{k}`: {v:.4f}" if isinstance(v, float) else f"- `{k}`: {v}")
        new_lines.append("")
        new_lines.append("</details>")

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
