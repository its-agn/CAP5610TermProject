"""Live updating elapsed timer for long running steps."""

import contextlib
import sys
import threading
import time
from collections.abc import Generator

MIN_LABEL_WIDTH = 40

@contextlib.contextmanager
def timed_step(label) -> Generator[None, None, None]:
    """Print a label with a live updating elapsed timer.

    Usage:
        with timed_step("Fitting model"):
            model.fit(X, y)
        # prints: Fitting model...                     (12.3s)
    """
    width = max(MIN_LABEL_WIDTH, len(label) + 6)
    padded = f"{label}...".ljust(width)
    out = sys.stdout
    start = time.monotonic()
    stop = threading.Event()

    def tick():
        while not stop.wait(0.1):
            if stop.is_set():
                return
            elapsed = time.monotonic() - start
            out.write(f"\r{padded} ({elapsed:.1f}s)")
            out.flush()

    ticker = threading.Thread(target=tick, daemon=True)
    ticker.start()
    try:
        yield
    finally:
        stop.set()
        ticker.join()
        elapsed = time.monotonic() - start
        out.write(f"\r{padded} ({elapsed:.1f}s)\n")
        out.flush()
