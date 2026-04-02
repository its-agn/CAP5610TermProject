"""Shared Optuna-based hyperparameter tuning for all models."""

import json
import os
import time
from datetime import datetime
from typing import Callable

def tune_model(
    objective: Callable,
    n_trials: int = 30,
    log_path: str | None = None,
    best_params_path: str | None = None,
    model_name: str = "Model",
):
    """Run Optuna hyperparameter optimization.

    Args:
        objective: function that takes an optuna.Trial, calls trial.suggest_*
                   for each hyperparameter, trains + evaluates, and returns
                   the score to maximize (e.g. macro F1).
        n_trials: number of trials to run.
        log_path: if set, write a markdown tuning log here (updated each trial).
        best_params_path: if set, save best params JSON here at the end.
        model_name: name for the tuning log header.

    Returns:
        dict with "best_params", "best_score", and "all_results"
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    print(f"{n_trials} trials to run\n")

    trial_count = [0]

    def timed_objective(trial):
        trial_count[0] += 1
        start = time.monotonic()
        print(f"Trial {trial_count[0]}/{n_trials}")
        score = objective(trial)
        elapsed = time.monotonic() - start
        print(f"  -> F1: {score:.4f} | Best so far: "
              f"{max(score, study.best_value if study.trials else 0):.4f} | {elapsed:.0f}s")
        print(f"     {trial.params}")
        print("-" * 60)
        return score

    def callback(_study, _trial):
        if _trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if log_path:
            write_tuning_log(_study_to_results(_study), log_path, model_name=model_name)

    study.optimize(timed_objective, n_trials=n_trials, callbacks=[callback])

    results = _study_to_results(study)

    print(f"\n{'='*60}")
    print(f"Overall best F1: {results['best_score']:.4f}")
    print(f"Best params: {results['best_params']}")
    print("=" * 60)

    if best_params_path:
        save_best_params(results["best_params"], best_params_path)

    return results

def _study_to_results(study):
    """Convert an Optuna study to our standard results dict."""
    import optuna

    all_results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            elapsed = (trial.datetime_complete - trial.datetime_start).total_seconds()
            all_results.append({
                "params": trial.params,
                "best_score": trial.value,
                "elapsed": elapsed,
            })

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "all_results": all_results,
    }

def save_best_params(params, path):
    """Save best params dict to JSON. Converts tuples to lists for serialization."""
    serializable = {}
    for k, v in params.items():
        serializable[k] = list(v) if isinstance(v, tuple) else v
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Best params saved to {path}")

def load_best_params(path):
    """Load best params from JSON. Converts lists back to tuples."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        params = json.load(f)
    for k, v in params.items():
        if isinstance(v, list):
            params[k] = tuple(v)
    return params

def write_tuning_log(results, path, model_name="Model"):
    """Write a markdown summary of tuning results.

    Columns are derived from the param keys in the results, so this works
    for any model without hardcoding column names.
    """
    if not results["all_results"]:
        return

    param_keys: list[str] = []
    for r in results["all_results"]:
        for k in r["params"]:
            if k not in param_keys:
                param_keys.append(k)

    def col_name(key):
        return key.replace("_", " ").title()

    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    headers = ["#"] + [col_name(k) for k in param_keys] + ["Macro F1", "Time (s)"]
    sep = "|".join("---" for _ in headers)

    with open(path, "w") as f:
        f.write(f"# {model_name} - Tuning Results ({date})\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + sep + "|\n")
        for i, r in enumerate(results["all_results"]):
            p = r["params"]
            vals = [str(i + 1)]
            for k in param_keys:
                v = p.get(k, "")
                if isinstance(v, (tuple, list)):
                    v = "/".join(str(x) for x in v)
                elif v is None:
                    v = "None"
                vals.append(str(v))
            vals.append(f"{r['best_score']:.4f}")
            vals.append(f"{r['elapsed']:.0f}")
            f.write("| " + " | ".join(vals) + " |\n")
    print(f"Tuning log written to {path}")
