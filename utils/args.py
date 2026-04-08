"""Common CLI arguments shared across all model scripts."""

import argparse

def common_parser():
    """Return an ArgumentParser preloaded with shared flags (--tune, --final, --discard)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true",
                        help="Run hyperparameter tuning")
    parser.add_argument("--final", action="store_true",
                        help="Full training set, evaluate on test")
    parser.add_argument("--discard", action="store_true",
                        help="Skip saving results (results_log.md and tuning outputs)")
    parser.add_argument("--default", action="store_true",
                        help="Use the default config even when tuned best_config.json exists")
    return parser
