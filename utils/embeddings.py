"""Pretrained embedding loading utilities."""

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
        "member_template": "glove.6B.{dim}d.txt",
        "local_template": "glove.6B.{dim}d.txt",
        "label": "GloVe 6B",
        "size": "~862MB",
    },
    "42B": {
        "url": "https://nlp.stanford.edu/data/glove.42B.300d.zip",
        "file": "glove.42B.300d.zip",
        "dims": [300],
        "format": "zip",
        "member_template": "glove.42B.300d.txt",
        "local_template": "glove.42B.300d.txt",
        "label": "GloVe 42B",
        "size": "~1.75GB",
    },
    "2024WG": {
        "url": "https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.300d.zip",
        "file": "glove.2024.wikigiga.300d.zip",
        "dims": [300],
        "format": "zip",
        "member_template": "wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt",
        "local_template": "glove.2024.wikigiga.300d.txt",
        "label": "GloVe 2024 WikiGigaword",
        "size": "~1.6GB",
    },
    "FT-WIKI-SUBWORD": {
        "url": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip",
        "file": "wiki-news-300d-1M-subword.vec.zip",
        "dims": [300],
        "format": "zip",
        "member_template": "wiki-news-300d-1M-subword.vec",
        "local_template": "wiki-news-300d-1M-subword.vec",
        "label": "fastText wiki-news subword",
        "size": "~1.0GB",
        "skip_header": True,
    },
}

def embedding_display_name(source="42B", dim=300):
    """Return a user-facing embedding name."""
    src = GLOVE_SOURCES[source]
    return f"{src['label']} {dim}d"

def _download(url, dest):
    """Download a file with progress indicator."""
    src = next((cfg for cfg in GLOVE_SOURCES.values() if cfg["url"] == url), None)
    size_label = src["size"] if src is not None else "unknown size"
    label = f"Downloading {os.path.basename(dest)} ({size_label})"
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
    """Download and extract an embedding file if needed."""
    src = GLOVE_SOURCES[source]
    if dim not in src["dims"]:
        raise ValueError(f"{src['label']} only supports dims {src['dims']}, got {dim}")
    member_name = src["member_template"].format(dim=dim)
    local_name = src.get("local_template", src["member_template"]).format(dim=dim)
    filepath = os.path.join(DATA_DIR, local_name)
    if os.path.exists(filepath):
        return filepath
    os.makedirs(DATA_DIR, exist_ok=True)
    archive_path = os.path.join(DATA_DIR, src["file"])
    if not os.path.exists(archive_path):
        _download(src["url"], archive_path)
    with timed_step(f"Extracting {local_name}"):
        with zipfile.ZipFile(archive_path) as zf:
            extracted_path = zf.extract(member_name, DATA_DIR)
            if extracted_path != filepath:
                os.replace(extracted_path, filepath)
    return filepath

def build_embedding_matrix(vocab, source="42B", dim=300):
    """Build an embedding matrix using pretrained vectors for words in vocab.

    Words not in the pretrained set get random initialization.
    Returns a numpy array of shape (vocab_size, dim).
    """
    src = GLOVE_SOURCES[source]
    filepath = _ensure_glove_file(source, dim)
    vocab_set = set(vocab.keys())
    vectors = {}
    with timed_step(f"Loading {embedding_display_name(source, dim)} vectors"):
        with open(filepath, encoding="utf-8") as f:
            if src.get("skip_header"):
                next(f, None)
            for line in f:
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                word = parts[0].lower()
                if word in vocab_set and word not in vectors:
                    vectors[word] = np.array([float(x) for x in parts[1].split()], dtype=np.float32)
    matrix = np.random.normal(scale=0.6, size=(len(vocab), dim)).astype(np.float32)
    matrix[0] = 0  # <pad> stays zero
    found = 0
    for word, idx in vocab.items():
        if word in vectors:
            matrix[idx] = vectors[word]
            found += 1
    print(f"Embedding coverage: {found}/{len(vocab)} ({100 * found / len(vocab):.0f}%)")
    return matrix
