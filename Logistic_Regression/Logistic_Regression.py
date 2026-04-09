import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    compute_metrics,
    load_best_config,
    load_yelp_data,
    plot_confusion_matrix,
    print_metrics,
    save_best_config,
    save_results,
    timed_step,
    tune_model,
)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_PARAMS_FILE = os.path.join(SCRIPT_DIR, "best_params.json")
TUNING_LOG = os.path.join(SCRIPT_DIR, "tuning_log.md")
RESULTS_LOG = os.path.join(SCRIPT_DIR, "results_log.md")


def extract_features(
    train_texts,
    eval_texts,
    max_features=40000,
    ngram_range=(1, 2),
    min_df=2,
):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    return vectorizer.fit_transform(train_texts), vectorizer.transform(eval_texts)


def train_model(
    X_train,
    y_train,
    C=4.0,
    penalty="l2",
    l1_ratio=None,
    max_iter=300,
    tol=1e-4,
    random_state=0,
):
    model_kwargs = {
        "C": C,
        "penalty": penalty,
        "solver": "saga",
        "max_iter": max_iter,
        "tol": tol,
        "random_state": random_state,
        "verbose": 0,
    }
    if penalty == "elasticnet":
        model_kwargs["l1_ratio"] = 0.5 if l1_ratio is None else l1_ratio

    model = LogisticRegression(**model_kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_eval, y_eval, verbose=True, model_name="Logistic Regression"):
    y_pred = model.predict(X_eval)
    metrics = compute_metrics(y_eval, y_pred)
    if verbose:
        print_metrics(metrics, model_name, y_eval, y_pred)
    return y_pred, metrics


def run_tuning():
    with timed_step("Loading dataset"):
        train_texts, y_train, val_texts, y_val, _, _ = load_yelp_data(
            train_size=100000,
            val_split=0.1,
        )

    assert val_texts is not None and y_val is not None
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    def objective(trial):
        tfidf_features = trial.suggest_categorical("tfidf_features", [20000, 40000, 60000])
        ngram_max = trial.suggest_int("ngram_max", 1, 2)
        min_df = trial.suggest_categorical("min_df", [1, 2, 5])
        C = trial.suggest_float("C", 0.25, 8.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l2", "elasticnet"])
        max_iter = trial.suggest_categorical("max_iter", [200, 300, 500])
        l1_ratio = (
            trial.suggest_float("l1_ratio", 0.05, 0.95)
            if penalty == "elasticnet"
            else None
        )

        X_train, X_val = extract_features(
            train_texts,
            val_texts,
            max_features=tfidf_features,
            ngram_range=(1, ngram_max),
            min_df=min_df,
        )
        model = train_model(
            X_train,
            y_train,
            C=C,
            penalty=penalty,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
        )
        _, metrics = evaluate(model, X_val, y_val, verbose=False)
        return metrics["macro_f1"]

    results = tune_model(
        objective,
        n_trials=30,
        log_path=TUNING_LOG,
        model_name="Logistic Regression",
    )
    save_best_config(results["best_config"], BEST_PARAMS_FILE)


def main(final=False):
    run_start = time.time()
    model_name = "Logistic Regression"

    best, _ = load_best_config(BEST_PARAMS_FILE)
    if best:
        print(f"Using tuned params: {best}")

    if final:
        with timed_step("Loading full dataset"):
            train_texts, y_train, _, _, eval_texts, y_eval = load_yelp_data(
                train_size=None,
                val_split=0,
            )
        eval_label = "TEST"
    else:
        with timed_step("Loading dataset"):
            train_texts, y_train, eval_texts, y_eval, _, _ = load_yelp_data(
                train_size=150000,
            )
        eval_label = "validation"

    assert eval_texts is not None and y_eval is not None
    print(f"Train: {len(train_texts)} | {eval_label}: {len(eval_texts)}")

    with timed_step("Extracting features"):
        X_train, X_eval = extract_features(
            train_texts,
            eval_texts,
            max_features=best.get("tfidf_features", 40000) if best else 40000,
            ngram_range=(1, best.get("ngram_max", 2)) if best else (1, 2),
            min_df=best.get("min_df", 2) if best else 2,
        )

    penalty = best.get("penalty", "l2") if best else "l2"
    l1_ratio = best.get("l1_ratio") if best and penalty == "elasticnet" else None

    with timed_step("Training model"):
        model = train_model(
            X_train,
            y_train,
            C=best.get("C", 4.0) if best else 4.0,
            penalty=penalty,
            l1_ratio=l1_ratio,
            max_iter=best.get("max_iter", 300) if best else 300,
        )

    with timed_step("Evaluating model"):
        y_pred, metrics = evaluate(model, X_eval, y_eval, model_name=model_name)
    cm_path = os.path.join(SCRIPT_DIR, "confusion_matrix_logreg.png")
    plot_confusion_matrix(y_eval, y_pred, cm_path, model_name)

    total = time.time() - run_start
    save_results(model_name, metrics, total, RESULTS_LOG, final=final)
    print(f"Done in {total:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--final", action="store_true", help="Full training set, evaluate on test")
    args = parser.parse_args()

    if args.tune:
        run_tuning()
    else:
        main(final=args.final)
