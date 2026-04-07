"""Shared dataset loading for Yelp Review Full."""

import contextlib

from .randomness import DEFAULT_SEED

LABEL_NAMES = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]

@contextlib.contextmanager
def _quiet_hf_loading():
    """Temporarily disable datasets progress bars while preserving prior state."""
    from datasets.utils import disable_progress_bars, enable_progress_bars, is_progress_bar_enabled
    from datasets.utils.logging import get_verbosity, set_verbosity, set_verbosity_error

    was_enabled = is_progress_bar_enabled()
    previous_verbosity = get_verbosity()
    if was_enabled:
        disable_progress_bars()
    set_verbosity_error()
    try:
        yield
    finally:
        set_verbosity(previous_verbosity)
        if was_enabled:
            enable_progress_bars()

def load_yelp_data(train_size: int | None = 150000, val_split: float = 0.1,
                   skip_test: bool = False, seed: int = DEFAULT_SEED):
    """Pull yelp_review_full from HuggingFace (cached after first download).

    Optionally subsample training set for shorter runs. Keeps class
    balance via stratified split. Carves out a validation set from the
    training data so the test set is not touched during tuning.
    Pass skip_test=True to load only the training split (saves time and memory during tuning).
    """
    import numpy as np
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    with _quiet_hf_loading():
        train_split = load_dataset("yelp_review_full", split="train")
        test_split = None if skip_test else load_dataset("yelp_review_full", split="test")

    train_labels_all = np.array(train_split["label"])
    train_indices = np.arange(len(train_split))
    if train_size and train_size < len(train_indices):
        train_indices, _ = train_test_split(
            train_indices,
            train_size=train_size,
            stratify=train_labels_all,
            random_state=seed,
        )

    val_indices = None
    if val_split and val_split > 0:
        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=val_split,
            stratify=train_labels_all[train_indices],
            random_state=seed,
        )

    train_subset = train_split.select(train_indices.tolist())
    train_texts = list(train_subset["text"])
    train_labels = np.array(train_subset["label"])

    val_texts, val_labels = None, None
    if val_indices is not None:
        val_subset = train_split.select(val_indices.tolist())
        val_texts = list(val_subset["text"])
        val_labels = np.array(val_subset["label"])

    if test_split is None:
        test_texts, test_labels = [], np.array([])
    else:
        test_texts = list(test_split["text"])
        test_labels = np.array(test_split["label"])

    return (
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels,
    )
