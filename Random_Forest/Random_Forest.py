"""
Random Forest on Yelp Review Full (5-class star rating prediction).
Keating Sane - CAP5610 Spring 2026

Run modes:
    python Random_Forest.py                      # quick run (subsampled, val set)
    python Random_Forest.py --single-tree        # same but with just a single Decision Tree
    python Random_Forest.py --tune               # hyperparameter grid search → tuning_log.md
    python Random_Forest.py --final              # full 650k train, evaluate on test set
    python Random_Forest.py --final --single-tree
"""
import argparse
import os
import sys
import time
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import timed_step

with timed_step("Loading libraries"):
    import numpy as np
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )
    import matplotlib.pyplot as plt
    import seaborn as sns

import logging
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.CRITICAL)

LABEL_NAMES = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TUNING_LOG = os.path.join(SCRIPT_DIR, "tuning_log.md")
RESULTS_LOG = os.path.join(SCRIPT_DIR, "results_log.md")

def load_yelp_data(train_size: int | None = 150000, val_split: float = 0.1):
    """
    Pull yelp_review_full from HuggingFace (cached after first download).
    Optionally subsample training set for shorter runs. Keeps class
    balance via stratified split. Carves out a validation set from the
    training data so the test set is not touched during tuning.
    """
    # silence HF warnings and progress bars so they do not interfere with timer
    _stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        dataset = load_dataset("yelp_review_full")
    finally:
        sys.stderr.close()
        sys.stderr = _stderr

    # convert to plain lists as HF returns lazy objects that are slow to index
    train_texts = list(dataset["train"]["text"])
    train_labels = list(dataset["train"]["label"])  # labels are 0-4, mapping to 1-5 stars
    test_texts = list(dataset["test"]["text"])
    test_labels = list(dataset["test"]["label"])

    # subsample if requested (full dataset is 650k which takes ~1hr for RF)
    if train_size and train_size < len(train_texts):
        train_texts, _, train_labels, _ = train_test_split(
            train_texts, train_labels,
            train_size=train_size, stratify=train_labels, random_state=0,
        )

    # hold out a validation set from training data
    val_texts, val_labels = None, None
    if val_split and val_split > 0:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels,
            test_size=val_split, stratify=train_labels, random_state=0,
        )

    return (
        train_texts, np.array(train_labels),
        val_texts, np.array(val_labels) if val_labels is not None else None,
        test_texts, np.array(test_labels),
    )

def extract_features(train_texts, eval_texts, max_features=20000, ngram_range=(1, 2)):
    """
    TF-IDF vectorization. Each review becomes a sparse vector of word/bigram weights.
    sublinear_tf uses log(1 + tf) which helps prevent long reviews from dominating.
    Vocabulary is learned from train_texts only, eval_texts just gets transformed.
    """
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
                min_samples_leaf=1, random_state=0, single_tree=False):
    if single_tree:
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        with timed_step("Fitting decision tree"):
            model.fit(X_train, y_train)
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,  # use all CPU cores
            random_state=random_state,
            verbose=0,
        )
        with timed_step(f"Fitting random forest ({n_estimators} trees)"):
            model.fit(X_train, y_train)
    return model

def evaluate(model, X_eval, y_eval, verbose=True, model_name="Random Forest"):
    """Run predictions and compute the evaluation metrics."""
    y_pred = model.predict(X_eval)

    metrics = {
        "accuracy": accuracy_score(y_eval, y_pred),
        "macro_precision": precision_score(y_eval, y_pred, average="macro"),
        "macro_recall": recall_score(y_eval, y_pred, average="macro"),
        "macro_f1": f1_score(y_eval, y_pred, average="macro"),
    }

    if verbose:
        print("=" * 60)
        print(f"{model_name} - Evaluation Results")
        print("=" * 60)
        print(f"Accuracy:        {metrics['accuracy']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
        print(f"Macro F1-Score:  {metrics['macro_f1']:.4f}")
        print()
        print(classification_report(y_eval, y_pred, target_names=LABEL_NAMES))

    return y_pred, metrics

def plot_confusion_matrix(y_eval, y_pred, save_path="confusion_matrix_rf.png",
                          model_name="Random Forest"):
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    cm = confusion_matrix(y_eval, y_pred)
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

def save_results(model_name, metrics, elapsed, final=False):
    """Update results_log.md with the latest run for this model type.

    Each model type (Decision Tree, Random Forest) gets its own section.
    Running again overwrites that section with the new results.
    """
    section = f"{model_name} ({'final' if final else 'validation'})"
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_lines = [
        f"## {section}",
        "",
        f"- **Date:** {date}",
        f"- **Accuracy:** {metrics['accuracy']:.4f}",
        f"- **Macro Precision:** {metrics['macro_precision']:.4f}",
        f"- **Macro Recall:** {metrics['macro_recall']:.4f}",
        f"- **Macro F1:** {metrics['macro_f1']:.4f}",
        f"- **Time:** {elapsed:.1f}s ({elapsed/60:.1f}m)",
    ]

    # parse existing sections from the log
    existing = {}
    if os.path.exists(RESULTS_LOG):
        with open(RESULTS_LOG) as f:
            current_key = None
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("## "):
                    current_key = line[3:]
                    existing[current_key] = []
                elif current_key is not None:
                    existing[current_key].append(line)

    # strip trailing blank lines from each section
    for key in existing:
        while existing[key] and existing[key][-1] == "":
            existing[key].pop()

    # update or add section
    existing[section] = new_lines[1:]  # skip the "## ..." line, key is the section name

    # write it all back
    with open(RESULTS_LOG, "w") as f:
        f.write("# Random Forest / Decision Tree - Results Log\n")
        for key, lines in existing.items():
            f.write(f"\n## {key}\n")
            for line in lines:
                f.write(f"{line}\n")

    print("Results saved to results_log.md")

def _init_tuning_log():
    """Create (or reset) the tuning log with a fresh table header."""
    with open(TUNING_LOG, "w") as f:
        f.write("# Random Forest - Hyperparameter Tuning Log\n\n")
        f.write("## Tuning decisions\n\n")
        f.write("Round 1 (54 combos) tested TF-IDF features [10k, 20k, 40k], "
                "ngrams [(1,1), (1,2)], estimators [50, 100, 200], "
                "max depth [50, 150, None]. Findings:\n")
        f.write("- 40k features never outperformed 20k (more noise, no gain)\n")
        f.write("- Bigrams (1,2) consistently beat unigrams (1,1) by 2-3%\n")
        f.write("- More trees always improved results (200 > 100 > 50)\n")
        f.write("- max_depth=50 was always worst; 150 vs None was negligible\n")
        f.write("- Best result: 20k features, (1,2), 200 trees, depth 150 -> 0.5300 Macro F1\n\n")
        f.write("Round 2 drops losers (40k features, unigrams, low tree counts, depth 50), "
                "adds trigrams (1,3), 300 trees, and min_samples_leaf regularization [1, 3, 5].\n\n")
        f.write("## Results\n\n")
        f.write("| # | Date | TF-IDF Features | Ngram | Estimators | Max Depth "
                "| Min Leaf | Train Size | Accuracy | Macro P | Macro R | Macro F1 | Time (s) |\n")
        f.write("|---|------|-----------------|-------|------------|-----------|"
                "----------|------------|----------|---------|---------|----------|----------|\n")

def _next_run_number():
    if not os.path.exists(TUNING_LOG):
        return 1
    count = 0
    with open(TUNING_LOG) as f:
        for line in f:
            # count data rows (skip header and separator lines)
            if line.startswith("|") and not line.startswith("| #") and not line.startswith("|--"):
                count += 1
    return count + 1

def log_tuning_result(params, metrics, elapsed):
    """Append one result row to tuning_log.md."""
    run = _next_run_number()
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    depth_str = str(params["max_depth"]) if params["max_depth"] else "None"
    ngram_str = f"{params['ngram_range'][0]}-{params['ngram_range'][1]}"

    leaf_str = str(params.get("min_samples_leaf", 1))
    row = (f"| {run} | {date} | {params['tfidf_features']} | {ngram_str} "
           f"| {params['n_estimators']} | {depth_str} | {leaf_str} | {params['train_size']} "
           f"| {metrics['accuracy']:.4f} | {metrics['macro_precision']:.4f} "
           f"| {metrics['macro_recall']:.4f} | {metrics['macro_f1']:.4f} "
           f"| {elapsed:.0f} |\n")

    with open(TUNING_LOG, "a") as f:
        f.write(row)
    print(f"  -> Logged as run #{run}")

# Grid of hyperparameters to search over.
# Each combo runs on 150k training samples to keep individual runs under ~10 min.
# Dropped 40k features and unigrams based on first tuning round results.
TUNING_GRID = {
    "tfidf_features": [10000, 20000],
    "ngram_range":    [(1, 2), (1, 3)],
    "n_estimators":   [200, 300],
    "max_depth":      [150, None],
    "min_samples_leaf": [1, 3, 5],
    "train_size":     [150000],
}

def run_tuning(grid=TUNING_GRID):
    """
    Tune on validation set only.
    """
    # reset tuning log for a fresh run
    _init_tuning_log()

    with timed_step("Loading dataset"):
        _stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        try:
            dataset = load_dataset("yelp_review_full")
        finally:
            sys.stderr.close()
            sys.stderr = _stderr
        all_train_texts = list(dataset["train"]["text"])
        all_train_labels = list(dataset["train"]["label"])
        del dataset  # free memory, only need the train split for tuning

    keys = list(grid.keys())
    combos = list(product(*grid.values()))
    print(f"{len(combos)} combinations to test")

    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{len(combos)}: {params}")
        print("=" * 60)

        # same stratified subsample every time so results are comparable
        train_texts, _, train_labels, _ = train_test_split(
            all_train_texts, all_train_labels,
            train_size=params["train_size"], stratify=all_train_labels, random_state=0,
        )

        # split off 10% as validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels,
            test_size=0.1, stratify=train_labels, random_state=0,
        )
        y_train = np.array(train_labels)
        y_val = np.array(val_labels)

        start = time.time()

        X_train, X_val = extract_features(
            train_texts, val_texts,
            max_features=params["tfidf_features"],
            ngram_range=params["ngram_range"],
        )

        model = train_model(
            X_train, y_train,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params.get("min_samples_leaf", 1),
        )

        _, metrics = evaluate(model, X_val, y_val, verbose=True)

        elapsed = time.time() - start
        print(f"  Time: {elapsed:.0f}s")
        log_tuning_result(params, metrics, elapsed)

    print(f"Finished. See all results in {TUNING_LOG}")

def main(single_tree=False, final=False):
    run_start = time.time()
    model_name = "Decision Tree" if single_tree else "Random Forest"

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

    X_train, X_eval = extract_features(train_texts, eval_texts)
    print(f"Feature matrix: {X_train.shape}")

    model = train_model(X_train, y_train, single_tree=single_tree)

    y_pred, metrics = evaluate(model, X_eval, y_eval, model_name=model_name)
    cm_path = "confusion_matrix_dt.png" if single_tree else "confusion_matrix_rf.png"
    plot_confusion_matrix(y_eval, y_pred, save_path=cm_path, model_name=model_name)

    total = time.time() - run_start
    save_results(model_name, metrics, total, final=final)
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f}m)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter grid search")
    parser.add_argument("--final", action="store_true", help="Full training set, evaluate on test")
    parser.add_argument("--single-tree", action="store_true",
                        help="Use a single Decision Tree instead of Random Forest")
    args = parser.parse_args()

    if args.tune:
        run_tuning()
    else:
        main(single_tree=args.single_tree, final=args.final)
