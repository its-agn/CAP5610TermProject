"""Live updating elapsed timer for long running steps."""

import contextlib
import sys
import threading
import time
from collections.abc import Generator

LINE_WIDTH = 64
MIN_LABEL_WIDTH = 24
TICK_INTERVAL = 0.1

@contextlib.contextmanager
def timed_step(label, suffix=None) -> Generator[None, None, None]:
    """Print a label with a live updating elapsed timer.

    Usage:
        with timed_step("Fitting model"):
            model.fit(X, y)
        # prints: Fitting model...                     (12.3s)
    """
    out = sys.stdout
    start = time.monotonic()
    stop = threading.Event()
    dot_count = [0]
    tick_count = [0]

    def render(final=False):
        dots = "" if final else "." * dot_count[0]
        elapsed = time.monotonic() - start
        timer_str = f"({elapsed:.1f}s)"
        suffix_str = f" {suffix()}" if suffix else ""
        content = f"{label}{dots}{suffix_str}"
        width = max(MIN_LABEL_WIDTH, LINE_WIDTH - len(timer_str) - 1)
        padded = content.ljust(width)
        end = "\n" if final else ""
        out.write(f"\r{padded} {timer_str}{end}")
        out.flush()

    def tick():
        while not stop.wait(TICK_INTERVAL):
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
