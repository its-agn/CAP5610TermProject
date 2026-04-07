"""GloVe embedding loading utilities."""

import os
import zipfile
import urllib.request

import numpy as np

from .timer import timed_step

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".data")

GLOVE_SOURCES = {
    "6B": {
        "url": "https://nlp.stanford.edu/data/glove.6B.zip",
        "file": "glove.6B.zip",
        "dims": [50, 100, 200, 300],
        "format": "zip",
    },
    "42B": {
        "url": "https://nlp.stanford.edu/data/glove.42B.300d.zip",
        "file": "glove.42B.300d.zip",
        "dims": [300],
        "format": "zip",
    },
}

def _download(url, dest):
    """Download a file with progress indicator."""
    size_mb = "~2GB" if "42B" in url else "~862MB"
    label = f"Downloading {os.path.basename(dest)} ({size_mb})"
    status = ["  0%"]
    def progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            status[0] = f"{pct:3d}%"
        else:
            mb = count * block_size / 1_000_000
            status[0] = f"{mb:.0f}MB"
    with timed_step(label, suffix=lambda: status[0]):
        urllib.request.urlretrieve(url, dest, progress)

def _ensure_glove_file(source="42B", dim=300):
    """Download and extract GloVe if needed. Returns path to the text file."""
    src = GLOVE_SOURCES[source]
    if dim not in src["dims"]:
        raise ValueError(f"GloVe {source} only supports dims {src['dims']}, got {dim}")
    filename = f"glove.{source}.{dim}d.txt"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return filepath
    os.makedirs(DATA_DIR, exist_ok=True)
    archive_path = os.path.join(DATA_DIR, src["file"])
    if not os.path.exists(archive_path):
        _download(src["url"], archive_path)
    with timed_step(f"Extracting {filename}"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extract(filename, DATA_DIR)
    return filepath

def build_embedding_matrix(vocab, source="42B", dim=300):
    """Build an embedding matrix using GloVe vectors for words in vocab.

    Words not in GloVe get random initialization.
    Returns a numpy array of shape (vocab_size, dim).
    """
    filepath = _ensure_glove_file(source, dim)
    vocab_set = set(vocab.keys())
    glove = {}
    with timed_step(f"Loading GloVe {source} {dim}d vectors"):
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                parts = line.split(" ", 1)
                word = parts[0].lower()
                if word in vocab_set and word not in glove:
                    glove[word] = np.array([float(x) for x in parts[1].split()], dtype=np.float32)
    matrix = np.random.normal(scale=0.6, size=(len(vocab), dim)).astype(np.float32)
    matrix[0] = 0  # <pad> stays zero
    found = 0
    for word, idx in vocab.items():
        if word in glove:
            matrix[idx] = glove[word]
            found += 1
    print(f"GloVe coverage: {found}/{len(vocab)} ({100 * found / len(vocab):.0f}%)")
    return matrix
