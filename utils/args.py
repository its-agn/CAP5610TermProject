"""Common CLI arguments shared across all model scripts."""

import argparse

def common_parser():
    """Return an ArgumentParser preloaded with shared flags (--tune, --final, --no-save)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true",
                        help="Run hyperparameter tuning")
    parser.add_argument("--final", action="store_true",
                        help="Full training set, evaluate on test")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving results to results_log.md")
    return parser
