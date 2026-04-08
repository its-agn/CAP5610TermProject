"""Shared utilities for all model scripts."""

import logging

# Suppress the huggingface_hub token-auth warning that fires
# whenever load_dataset() is called without an authenticated session.
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.CRITICAL)

from .args import common_parser
from .data import LABEL_NAMES, load_yelp_data
from .evaluation import (
    compute_metrics,
    get_device_name,
    plot_confusion_matrix,
    print_metrics,
    print_run_header,
    print_value_section,
    save_results,
)
from .embeddings import build_embedding_matrix, embedding_display_name
from .randomness import DEFAULT_SEED, set_random_seed
from .text_features import fit_tfidf_features
from .timer import timed_step
from .tuning import load_best_config, save_best_config, tune_model, write_tuning_log

__all__ = [
    "LABEL_NAMES",
    "build_embedding_matrix",
    "common_parser",
    "compute_metrics",
    "DEFAULT_SEED",
    "embedding_display_name",
    "fit_tfidf_features",
    "get_device_name",
    "load_best_config",
    "load_yelp_data",
    "plot_confusion_matrix",
    "print_metrics",
    "print_run_header",
    "print_value_section",
    "save_best_config",
    "save_results",
    "set_random_seed",
    "timed_step",
    "tune_model",
    "write_tuning_log",
]
