"""Shared dataset loading for Yelp Review Full."""

import sys
import os

LABEL_NAMES = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]

def load_yelp_data(train_size: int | None = 150000, val_split: float = 0.1,
                   skip_test: bool = False):
    """Pull yelp_review_full from HuggingFace (cached after first download).

    Optionally subsample training set for shorter runs. Keeps class
    balance via stratified split. Carves out a validation set from the
    training data so the test set is not touched during tuning.
    Pass skip_test=True to skip loading the test split (saves time during tuning).
    """
    import numpy as np
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    _stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        dataset = load_dataset("yelp_review_full")
    finally:
        sys.stderr.close()
        sys.stderr = _stderr

    train_texts = list(dataset["train"]["text"])
    train_labels = list(dataset["train"]["label"])
    if skip_test:
        test_texts, test_labels = [], []
    else:
        test_texts = list(dataset["test"]["text"])
        test_labels = list(dataset["test"]["label"])

    if train_size and train_size < len(train_texts):
        train_texts, _, train_labels, _ = train_test_split(
            train_texts, train_labels,
            train_size=train_size, stratify=train_labels, random_state=0,
        )

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
