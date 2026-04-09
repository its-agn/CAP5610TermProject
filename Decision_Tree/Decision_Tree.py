"""
Decision Tree on Yelp Review Full (5-class star rating prediction).
Keating Sane - CAP5610 Spring 2026

Usage: python Decision_Tree.py [common flags]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    DEFAULT_SEED,
    common_parser,
    compute_metrics,
    fit_tfidf_features,
    get_device_name,
    load_best_config,
    load_yelp_data,
    plot_confusion_matrix,
    print_metrics,
    print_run_header,
    print_value_section,
    save_best_config,
    save_results,
    set_random_seed,
    timed_step,
    tune_model,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TUNING_LOG = os.path.join(SCRIPT_DIR, "tuning_log.md")
RESULTS_LOG = os.path.join(SCRIPT_DIR, "results_log.md")
BEST_CONFIG_FILE = os.path.join(SCRIPT_DIR, "best_config.json")
TUNING_TRAIN_SIZE = 200000
TUNING_VAL_SPLIT = 0.1
TUNING_TRIALS = 20

if __name__ == "__main__":
    parser = common_parser()
    args = parser.parse_args()
    if args.tune and (args.final or args.default):
        parser.error("--tune cannot be combined with --final or --default")

with timed_step("Loading libraries"):
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

def train_model(X_train, y_train, max_depth=100, min_samples_leaf=5,
                random_state=DEFAULT_SEED, indent=""):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    with timed_step(f"{indent}Fitting decision tree"):
        model.fit(X_train, y_train)
    return model

def evaluate(model, X_eval, y_eval, verbose=True, model_name="Decision Tree", indent=""):
    """Run predictions and compute evaluation metrics."""
    with timed_step(f"{indent}Running inference"):
        y_pred = model.predict(X_eval)
    metrics = compute_metrics(y_eval, y_pred)
    if verbose:
        print_metrics(metrics, model_name, y_eval, y_pred)
    return y_pred, metrics

def run_tuning(discard=False):
    """Tune Decision Tree with Optuna."""
    set_random_seed(DEFAULT_SEED)
    print_run_header(
        "Decision Tree",
        mode="tuning",
        device=get_device_name(cpu_only=True),
        extra_info={
            "Dataset": f"{TUNING_TRAIN_SIZE} train ({int(TUNING_VAL_SPLIT * 100)}% val split)",
            "Trials": TUNING_TRIALS,
        },
    )

    with timed_step("Loading dataset"):
        train_texts, train_labels, val_texts, val_labels, _, _ = load_yelp_data(
            train_size=TUNING_TRAIN_SIZE,
            val_split=TUNING_VAL_SPLIT,
            skip_test=True,
            seed=DEFAULT_SEED,
        )
    assert val_texts is not None and val_labels is not None
    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    def objective(trial):
        tfidf_features = trial.suggest_categorical("tfidf_features", [5000, 10000, 20000, 40000])
        ngram_max = trial.suggest_int("ngram_max", 1, 2)
        min_df = trial.suggest_categorical("min_df", [3, 5, 10])
        max_df = trial.suggest_categorical("max_df", [0.85, 0.9, 0.95])
        max_depth = trial.suggest_categorical("max_depth", [50, 100, 150, 300, None])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

        X_train, X_val = fit_tfidf_features(
            train_texts,
            val_texts,
            max_features=tfidf_features,
            ngram_range=(1, ngram_max),
            min_df=min_df,
            max_df=max_df,
            indent="  ",
        )
        model = train_model(
            X_train,
            y_train,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=DEFAULT_SEED,
            indent="  ",
        )
        _, metrics = evaluate(model, X_val, y_val, verbose=False, indent="  ")
        return metrics["macro_f1"]

    results = tune_model(
        objective,
        n_trials=TUNING_TRIALS,
        log_path=None if discard else TUNING_LOG,
        model_name="Decision Tree",
        cpu_only=True,
        extra_info={
            "Dataset": f"{TUNING_TRAIN_SIZE} train ({int(TUNING_VAL_SPLIT * 100)}% val split)",
            "Trials": TUNING_TRIALS,
        },
        seed=DEFAULT_SEED,
    )
    if not discard:
        save_best_config(
            results["best_config"],
            BEST_CONFIG_FILE,
            metadata={
                "seed": DEFAULT_SEED,
                "tuning_train_size": len(train_texts) + len(val_texts),
                "tuning_val_split": TUNING_VAL_SPLIT,
                "tuning_trials": TUNING_TRIALS,
            },
            macro_f1=results["best_score"],
        )

def main(final=False, discard=False, default_config=False):
    set_random_seed(DEFAULT_SEED)
    run_start = time.monotonic()
    model_name = "Decision Tree"

    DEFAULT_PARAMS = {
        "tfidf_features": 20000,
        "ngram_max": 2,
        "min_df": 5,
        "max_df": 0.9,
        "max_depth": 100,
        "min_samples_leaf": 5,
    }

    params, metadata = (None, {}) if default_config else load_best_config(BEST_CONFIG_FILE)
    using_defaults = params is None
    if params is None:
        params = DEFAULT_PARAMS
    params_source = "default config" if using_defaults else "best tuned config"

    print_run_header(
        model_name,
        mode="final" if final else "validation",
        device=get_device_name(cpu_only=True),
        extra_info={"Config source": params_source},
    )
    print_value_section("Parameters", params)

    if final:
        with timed_step("Loading full dataset (650k, no subsampling)"):
            train_texts, y_train, _, _, eval_texts, y_eval = load_yelp_data(
                train_size=None, val_split=0, seed=DEFAULT_SEED,
            )
        eval_label = "TEST"
    else:
        with timed_step("Loading dataset (subsampled to 150k)"):
            train_texts, y_train, eval_texts, y_eval, _, _ = load_yelp_data(
                train_size=150000,
                seed=DEFAULT_SEED,
            )
        eval_label = "validation"

    assert eval_texts is not None and y_eval is not None
    print(f"Train: {len(train_texts)} | {eval_label}: {len(eval_texts)}")

    X_train, X_eval = fit_tfidf_features(
        train_texts,
        eval_texts,
        max_features=params["tfidf_features"],
        ngram_range=(1, params["ngram_max"]),
        min_df=params["min_df"],
        max_df=params["max_df"],
    )
    print(f"Feature matrix: {X_train.shape}")

    model = train_model(
        X_train,
        y_train,
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=DEFAULT_SEED,
    )

    y_pred, metrics = evaluate(model, X_eval, y_eval, model_name=model_name)

    total = time.monotonic() - run_start
    if not discard:
        device = get_device_name(cpu_only=True)
        saved = save_results(
            model_name,
            metrics,
            total,
            RESULTS_LOG,
            final=final,
            device=device,
            default_config=using_defaults,
            params=params,
            metadata=({"seed": DEFAULT_SEED} if using_defaults else metadata),
        )
        if final and not default_config and saved:
            cm_path = os.path.join(SCRIPT_DIR, "confusion_matrix_dt.png")
            plot_confusion_matrix(
                y_eval,
                y_pred,
                cm_path,
                model_name,
                title_suffix=f"Macro F1={metrics['macro_f1']:.4f}",
            )
        elif final and not default_config:
            print("Keeping existing confusion matrix because results were not updated")
    else:
        print("Skipping save (--discard)")
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f}m)")

if __name__ == "__main__":
    if args.tune:
        run_tuning(discard=args.discard)
    else:
        main(final=args.final, discard=args.discard, default_config=args.default)
