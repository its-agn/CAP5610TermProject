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
    out = sys.stdout
    start = time.monotonic()
    stop = threading.Event()
    dot_count = [0]
    tick_count = [0]

    def tick():
        while not stop.wait(0.1):
            tick_count[0] += 1
            if tick_count[0] % 2 == 0:
                dot_count[0] = (dot_count[0] % 3) + 1
            dots = "." * dot_count[0]
            padded = f"{label}{dots}".ljust(width)
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
        padded = label.ljust(width)
        out.write(f"\r{padded} ({elapsed:.1f}s)\n")
        out.flush()
