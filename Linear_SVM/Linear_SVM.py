'''
Linear Support Vector Machine (SVM) on YelpReviewFull dataset (5-class star rating prediction).

Run modes:
    python Linear_SVM.py              # trains on 10k samples
    python Linear_SVM.py --size n     # trains on n samples
    python Linear_SVM.py --final      # runs on full test set
'''
import argparse
import os
import sys
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import (
    DEFAULT_SEED,
    timed_step,
    load_yelp_data,
    compute_metrics,
    print_metrics,
    plot_confusion_matrix,
    save_results,
    tune_model,
    save_best_config,
    load_best_config,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TUNING_LOG = os.path.join(SCRIPT_DIR, "tuning_log.md")
BEST_PARAMS_FILE = os.path.join(SCRIPT_DIR, "best_params.json")


def run_tuning():
    train_texts, y_train, val_texts, y_val, _, _ = load_yelp_data(
        train_size=10000, val_split=0.2
    )

    def objective(trial):
        c_val = trial.suggest_float("C", 0.1, 10.0, log=True)
        max_feats = trial.suggest_categorical("max_features", [5000, 10000, 15000])

        vectorizer = TfidfVectorizer(max_features=max_feats, stop_words="english")
        X_train = vectorizer.fit_transform(train_texts)
        X_val = vectorizer.transform(val_texts)

        model = LinearSVC(C=c_val, max_iter=10000, random_state=DEFAULT_SEED)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        metrics = compute_metrics(y_val, y_pred)
        return metrics["macro_f1"]

    results = tune_model(
        objective,
        n_trials=15,
        log_path=TUNING_LOG,
        model_name="Linear SVM",
    )
    save_best_config(results["best_config"], BEST_PARAMS_FILE)
    return results


def main():
    start_time = time.time()

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run automated Optuna hyperparameter tuning")
    parser.add_argument("--size", type=int, default=10000, help="Number of training samples (default: 10000)")
    parser.add_argument("--final", action="store_true", help="Evaluate on the full test set")
    args = parser.parse_args()

    # tune
    if args.tune:
        print("Running automated hyperparameter tuning... (this may take a while)")
        run_tuning()
        print(f"Tuning complete! Best params saved to {BEST_PARAMS_FILE}")
        return

    # load best params from tune
    best, _ = load_best_config(BEST_PARAMS_FILE)

    #  load sub-sampled dataset
    with timed_step(f"Loading Dataset (size={args.size})"):
        train_texts, y_train, val_texts, y_val, test_texts, test_labels = load_yelp_data(train_size=args.size)

        final_string = None
        if args.final:
            val_texts, y_val = test_texts, test_labels
            final_string = "(full test set)"

    # Vectorize to TF-IDF for running
    with timed_step("Running TF-IDF"):
        vectorizer = TfidfVectorizer(
            max_features=best.get("max_features", 10000) if best else 10000,
            stop_words="english",
        )
        X_train_vectors = vectorizer.fit_transform(train_texts)
        X_val_vectors = vectorizer.transform(val_texts)

    # Train model
    C = best.get("C", 1) if best else 1
    with timed_step(f"Training Linear SVM (C={C})"):
        svm_model = LinearSVC(C=C, max_iter=10000, random_state=DEFAULT_SEED)
        svm_model.fit(X_train_vectors, y_train)

    # Evaluate
    with timed_step(
        f"Making predictions and evaluating {final_string if final_string is not None else ''}"
    ):
        y_pred = svm_model.predict(X_val_vectors)
        metrics = compute_metrics(y_val, y_pred)
        print_metrics(metrics, model_name="Linear SVM", y_true=y_val, y_pred=y_pred)

    results_file = os.path.join(SCRIPT_DIR, "results_log.md")
    cm_file = os.path.join(SCRIPT_DIR, "confusion_matrix_linear_svm.png")

    plot_confusion_matrix(y_val, y_pred, cm_file, "Linear SVM")

    total_time = time.time() - start_time
    save_results("Linear SVM", metrics, total_time, results_file, final=args.final)


if __name__ == "__main__":
    main()