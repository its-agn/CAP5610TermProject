"""Live updating elapsed timer for long running steps."""

import contextlib
import sys
import threading
import time
from collections.abc import Generator

MIN_LABEL_WIDTH = 40

@contextlib.contextmanager
def timed_step(label, suffix=None) -> Generator[None, None, None]:
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

    def render(final=False):
        dots = "" if final else "." * dot_count[0]
        padded = f"{label}{dots}".ljust(width)
        elapsed = time.monotonic() - start
        suffix_str = f" {suffix()}" if suffix else ""
        end = "\n" if final else ""
        out.write(f"\r{padded}{suffix_str} ({elapsed:.1f}s){end}")
        out.flush()

    def tick():
        while not stop.wait(0.1):
            tick_count[0] += 1
            if tick_count[0] % 2 == 0:
                dot_count[0] = (dot_count[0] % 3) + 1
            render()

    ticker = threading.Thread(target=tick, daemon=True)
    ticker.start()
    try:
        yield
    finally:
        stop.set()
        ticker.join()
        render(final=True)
