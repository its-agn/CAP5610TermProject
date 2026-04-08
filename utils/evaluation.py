"""Shared evaluation, plotting, and results logging."""

import json
import os
import platform
from datetime import datetime

from .data import LABEL_NAMES
from .timer import timed_step

HEADER_WIDTH = 64

def _format_display_value(value):
    """Format values for human-facing console/log display."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

def print_value_section(title, values):
    """Print an indented key/value section for runtime summaries."""
    if not values:
        return
    print(f"{title}:")
    for key, value in values.items():
        print(f"  {key}: {_format_display_value(value)}")

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

def print_run_header(title, mode=None, device=None, seed=None, extra_info=None):
    """Print a consistent header for training and tuning runs."""
    print("=" * HEADER_WIDTH)
    print(title if mode is None else f"{title} - {mode.title()} Run")
    print("=" * HEADER_WIDTH)
    if device is not None:
        print(f"Device: {device}")
    if seed is not None:
        print(f"Seed:   {seed}")
    if extra_info:
        for key, value in extra_info.items():
            print(f"{key}: {value}")

def compute_metrics(y_true, y_pred):
    """Compute standard multi-class classification metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

def print_metrics(metrics, model_name, y_true=None, y_pred=None):
    """Print formatted evaluation results. Includes classification_report if y_true/y_pred given."""
    from sklearn.metrics import classification_report

    print("=" * HEADER_WIDTH)
    print(f"{model_name} - Evaluation Results")
    print("=" * HEADER_WIDTH)
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score:  {metrics['macro_f1']:.4f}")
    print()
    if y_true is not None and y_pred is not None:
        print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0))

def plot_confusion_matrix(y_true, y_pred, save_path, model_name, title_suffix=None):
    """Save a confusion matrix heatmap to disk."""
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    with timed_step(f"Saving confusion matrix ({os.path.basename(save_path)})"):
        date = datetime.now().strftime("%Y-%m-%d %H:%M")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        title = model_name
        if title_suffix:
            title += f" | {title_suffix}"
        title += f" | {date}"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    print(f"Confusion matrix saved to {save_path}")

def _result_section_name(model_name, final=False, default_config: bool | None = False):
    """Return the results_log.md section header for a given run configuration."""
    mode = "final" if final else "validation"
    if default_config is None:
        return f"{model_name} ({mode})"
    config_tag = "default config" if default_config else "best config"
    return f"{model_name} ({mode}, {config_tag})"

def _get_logged_macro_f1(log_path, section):
    """Read the existing displayed Macro F1 for a section, if present."""
    if not os.path.exists(log_path):
        return None
    in_section = False
    with open(log_path) as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line.startswith("## "):
                in_section = line[3:] == section
                continue
            if not in_section:
                continue
            if "Macro F1" in line:
                try:
                    return float(line.rsplit(" ", 1)[1])
                except (IndexError, ValueError):
                    return None
    return None

def _should_update_result(log_path, model_name, new_macro_f1, final=False,
                          default_config: bool | None = False):
    """Return whether a run should overwrite the saved result for its section."""
    section = _result_section_name(
        model_name,
        final=final,
        default_config=default_config,
    )
    old_macro_f1 = _get_logged_macro_f1(log_path, section)
    return old_macro_f1 is None or new_macro_f1 >= old_macro_f1, section, old_macro_f1

def save_results(model_name, metrics, elapsed, log_path, final=False, device=None,
                 default_config: bool | None = False, params=None, metadata=None,
                 extra_info=None, results_name=None):
    """Update a results_log.md with the latest run for this model.

    Each model/mode combo (e.g. "Random Forest (final)") gets its own section.
    Running again overwrites that section with new results.
    Pass default_config=None to omit the config tag.
    """
    section_model_name = results_name or model_name
    should_update, section, old_macro_f1 = _should_update_result(
        log_path,
        section_model_name,
        metrics["macro_f1"],
        final=final,
        default_config=default_config,
    )
    if not should_update:
        print(
            f"Keeping existing results for '{section}' "
            f"(existing {old_macro_f1:.4f} > new {metrics['macro_f1']:.4f})"
        )
        return False

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
    if extra_info:
        for key, value in extra_info.items():
            new_lines.append(f"- **{key}:** {value}")
    if params or metadata:
        new_lines.append("")
        new_lines.append("<details>")
        new_lines.append("<summary>Config</summary>")
        new_lines.append("")
        if params:
            new_lines.append("**Params**")
            for k, v in params.items():
                new_lines.append(f"- `{k}`: {v}")
        if metadata:
            if params:
                new_lines.append("")
            new_lines.append("**Metadata**")
            for k, v in metadata.items():
                new_lines.append(f"- `{k}`: {v}")
        new_lines.append("")
        new_lines.append("</details>")

    with timed_step(f"Writing results log ({os.path.basename(log_path)})"):
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
    return True
