"""
Random Forest on Yelp Review Full (5-class star rating prediction).
Keating Sane - CAP5610 Spring 2026

Run modes:
    python Random_Forest.py                      # quick run (subsampled, val set)
    python Random_Forest.py --single-tree        # same but with just a single Decision Tree
    python Random_Forest.py --tune               # Optuna hyperparameter tuning → tuning_log.md
    python Random_Forest.py --final              # full 650k train, evaluate on test set
    python Random_Forest.py --final --single-tree
"""
import argparse
import os
import sys
import time
from typing import Literal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    compute_metrics,
    load_best_params,
    load_yelp_data,
    plot_confusion_matrix,
    print_metrics,
    save_results,
    timed_step,
    tune_model,
)

with timed_step("Loading libraries"):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

import logging
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.CRITICAL)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TUNING_LOG = os.path.join(SCRIPT_DIR, "tuning_log.md")
RESULTS_LOG = os.path.join(SCRIPT_DIR, "results_log.md")
BEST_PARAMS_FILE = os.path.join(SCRIPT_DIR, "best_params.json")

def extract_features(train_texts, eval_texts, max_features=20000, ngram_range=(1, 2)):
    """TF-IDF vectorization. Vocabulary is learned from train_texts only."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    with timed_step("Fitting TF-IDF on training data"):
        X_train = vectorizer.fit_transform(train_texts)
    with timed_step("Transforming eval data"):
        X_eval = vectorizer.transform(eval_texts)
    return X_train, X_eval

def train_model(X_train, y_train, n_estimators=100, max_depth=150,
                min_samples_leaf=1,
                max_features: Literal["sqrt", "log2"] | None = "sqrt",
                random_state=0, single_tree=False):
    if single_tree:
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
        )
        with timed_step("Fitting decision tree"):
            model.fit(X_train, y_train)
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features or "sqrt",
            n_jobs=-1,
            random_state=random_state,
            verbose=0,
        )
        with timed_step(f"Fitting random forest ({n_estimators} trees)"):
            model.fit(X_train, y_train)
    return model

def evaluate(model, X_eval, y_eval, verbose=True, model_name="Random Forest"):
    """Run predictions and compute evaluation metrics."""
    y_pred = model.predict(X_eval)
    metrics = compute_metrics(y_eval, y_pred)
    if verbose:
        print_metrics(metrics, model_name, y_eval, y_pred)
    return y_pred, metrics

def run_tuning():
    """Tune RF with Optuna (Bayesian optimization)."""
    from sklearn.model_selection import train_test_split

    with timed_step("Loading dataset"):
        all_texts, all_labels, _, _, _, _ = load_yelp_data(train_size=None, val_split=0)

    # subsample once for tuning
    train_texts, _, train_labels, _ = train_test_split(
        all_texts, list(all_labels),
        train_size=150000, stratify=all_labels, random_state=0,
    )
    y = np.array(train_labels)

    def objective(trial):
        # TF-IDF params
        tfidf_features = trial.suggest_categorical("tfidf_features", [10000, 20000, 40000])
        ngram_max = trial.suggest_int("ngram_max", 1, 2)

        # RF params
        n_estimators = trial.suggest_categorical("n_estimators", [50, 100, 200])
        max_depth = trial.suggest_categorical("max_depth", [50, 100, 150])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        max_features_choice = trial.suggest_categorical("max_features", ["sqrt", "log2"])

        vectorizer = TfidfVectorizer(
            max_features=tfidf_features,
            ngram_range=(1, ngram_max),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        X = vectorizer.fit_transform(train_texts)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features_choice,
            n_jobs=-1,
            random_state=0,
        )

        scores = cross_val_score(model, X, y, cv=3, scoring="f1_macro")
        return scores.mean()

    tune_model(
        objective,
        n_trials=30,
        log_path=TUNING_LOG,
        best_params_path=BEST_PARAMS_FILE,
        model_name="Random Forest",
    )

def main(single_tree=False, final=False):
    run_start = time.time()
    model_name = "Decision Tree" if single_tree else "Random Forest"

    best = load_best_params(BEST_PARAMS_FILE) if not single_tree else None
    if best:
        print("Using best tuned params (from best_params.json):")
        for k, v in best.items():
            print(f"  {k}: {v}")

    if final:
        with timed_step("Loading full dataset (650k, no subsampling)"):
            train_texts, y_train, _, _, eval_texts, y_eval = load_yelp_data(
                train_size=None, val_split=0,
            )
        eval_label = "TEST"
    else:
        with timed_step("Loading dataset (subsampled to 150k)"):
            train_texts, y_train, eval_texts, y_eval, _, _ = load_yelp_data(train_size=150000)
        eval_label = "validation"

    assert eval_texts is not None and y_eval is not None
    print(f"Train: {len(train_texts)} | {eval_label}: {len(eval_texts)}")

    X_train, X_eval = extract_features(
        train_texts, eval_texts,
        max_features=best.get("tfidf_features", 20000) if best else 20000,
        ngram_range=(1, best.get("ngram_max", 2)) if best else (1, 2),
    )
    print(f"Feature matrix: {X_train.shape}")

    model = train_model(
        X_train, y_train,
        n_estimators=best.get("n_estimators", 100) if best else 100,
        max_depth=best.get("max_depth", 150) if best else 150,
        min_samples_leaf=best.get("min_samples_leaf", 1) if best else 1,
        max_features=best.get("max_features", "sqrt") if best else ("sqrt" if not single_tree else None),
        single_tree=single_tree,
    )

    y_pred, metrics = evaluate(model, X_eval, y_eval, model_name=model_name)
    cm_name = "confusion_matrix_dt.png" if single_tree else "confusion_matrix_rf.png"
    cm_path = os.path.join(SCRIPT_DIR, cm_name)
    plot_confusion_matrix(y_eval, y_pred, cm_path, model_name)

    total = time.time() - run_start
    save_results(model_name, metrics, total, RESULTS_LOG, final=final)
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f}m)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--final", action="store_true", help="Full training set, evaluate on test")
    parser.add_argument("--single-tree", action="store_true",
                        help="Use a single Decision Tree instead of Random Forest")
    args = parser.parse_args()

    if args.tune:
        run_tuning()
    else:
        main(single_tree=args.single_tree, final=args.final)
