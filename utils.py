"""Shared utilities for all model scripts."""

import contextlib
import sys
import threading
import time

LABEL_WIDTH = 48

@contextlib.contextmanager
def timed_step(label):
    """Print a label with a live-updating elapsed timer.

    Usage:
        with timed_step("Fitting model"):
            model.fit(X, y)
        # prints: Fitting model...                     (12.3s)
    """
    padded = f"{label}...".ljust(LABEL_WIDTH)
    out = sys.stdout
    start = time.time()
    stop = threading.Event()

    # write label then save cursor position right after it
    # \033[s = save cursor, \033[u = restore cursor, \033[K = clear to end of line
    out.write(f"{padded}\033[s")
    out.flush()

    def tick():
        while not stop.wait(0.1):
            if stop.is_set():
                return
            elapsed = time.time() - start
            out.write(f"\033[u\033[K ({elapsed:.1f}s)")
            out.flush()

    ticker = threading.Thread(target=tick, daemon=True)
    ticker.start()
    try:
        yield
    finally:
        stop.set()
        ticker.join()
        elapsed = time.time() - start
        out.write(f"\033[u\033[K ({elapsed:.1f}s)\n")
        out.flush()
