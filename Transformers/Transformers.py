import argparse
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    compute_metrics,
    load_best_params,
    load_yelp_data,
    plot_confusion_matrix,
    print_metrics,
    save_results,
    timed_step,
    tune_model,
)

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_PARAMS_FILE = os.path.join(SCRIPT_DIR, "best_params.json")
TUNING_LOG = os.path.join(SCRIPT_DIR, "tuning_log.md")
RESULTS_LOG = os.path.join(SCRIPT_DIR, "results_log.md")
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
TEXT_RE = re.compile(r"[^\w\s]")


def clean_text(text):
    return TEXT_RE.sub(" ", text.lower())


def build_vocab(texts, max_vocab=30000):
    counter = Counter()
    for text in texts:
        counter.update(clean_text(text).split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def texts_to_indices(texts, vocab, max_len=128):
    indices = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        words = clean_text(text).split()[:max_len]
        for j, word in enumerate(words):
            indices[i, j] = vocab.get(word, 1)
    return indices


class EncoderClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len=128,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.2,
        num_classes=5,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        padding_mask = x.eq(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        pooled = self.norm(self.dropout(pooled))
        return self.classifier(pooled)


def build_model(vocab_size, params):
    return EncoderClassifier(
        vocab_size=vocab_size,
        max_len=params["max_len"],
        embed_dim=params["embed_dim"],
        num_heads=params["num_heads"],
        num_layers=params["num_layers"],
        ff_dim=params["ff_dim"],
        dropout=params["dropout"],
    )


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=3,
    batch_size=64,
    lr=3e-4,
    weight_decay=1e-4,
):
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_score = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_items = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0)
            total_items += batch_X.size(0)

        _, val_metrics = evaluate(
            model,
            X_val,
            y_val,
            batch_size=batch_size,
            verbose=False,
        )
        train_loss = total_loss / max(total_items, 1)
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"loss: {train_loss:.4f} - val_f1: {val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate(
    model,
    X_eval,
    y_eval,
    batch_size=128,
    verbose=True,
    model_name="Transformer Encoder",
):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_eval)),
        batch_size=batch_size,
        shuffle=False,
    )
    preds = []

    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(DEVICE)
            logits = model(batch_X)
            preds.append(logits.argmax(dim=1).cpu().numpy())

    y_pred = np.concatenate(preds)
    metrics = compute_metrics(y_eval, y_pred)
    if verbose:
        print_metrics(metrics, model_name, y_eval, y_pred)
    return y_pred, metrics


def get_default_params():
    return {
        "max_vocab": 30000,
        "max_len": 128,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "ff_dim": 256,
        "dropout": 0.2,
        "epochs": 3,
        "batch_size": 64,
        "lr": 3e-4,
        "weight_decay": 1e-4,
    }


def merge_params(best):
    params = get_default_params()
    if best:
        params.update(best)
    return params


def run_tuning():
    with timed_step("Loading dataset"):
        train_texts, y_train, val_texts, y_val, _, _ = load_yelp_data(
            train_size=20000,
            val_split=0.1,
        )

    assert val_texts is not None and y_val is not None
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    def objective(trial):
        params = {
            "max_vocab": trial.suggest_categorical("max_vocab", [15000, 25000, 30000]),
            "max_len": trial.suggest_categorical("max_len", [96, 128, 160]),
            "embed_dim": trial.suggest_categorical("embed_dim", [128, 256]),
            "num_heads": trial.suggest_categorical("num_heads", [4, 8]),
            "num_layers": trial.suggest_categorical("num_layers", [2, 3]),
            "ff_dim": trial.suggest_categorical("ff_dim", [256, 512]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.3),
            "epochs": trial.suggest_categorical("epochs", [2, 3]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
            "lr": trial.suggest_float("lr", 1e-4, 7e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        }

        vocab = build_vocab(train_texts, max_vocab=params["max_vocab"])
        X_train = texts_to_indices(train_texts, vocab, max_len=params["max_len"])
        X_val = texts_to_indices(val_texts, vocab, max_len=params["max_len"])

        model = build_model(len(vocab), params)
        model = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )
        _, metrics = evaluate(
            model,
            X_val,
            y_val,
            batch_size=params["batch_size"],
            verbose=False,
        )
        return metrics["macro_f1"]

    tune_model(
        objective,
        n_trials=10,
        log_path=TUNING_LOG,
        best_params_path=BEST_PARAMS_FILE,
        model_name="Transformer Encoder",
    )


def main(final=False):
    run_start = time.time()
    model_name = "Transformer Encoder"
    best = load_best_params(BEST_PARAMS_FILE)
    params = merge_params(best)

    if best:
        print(f"Using tuned params: {best}")

    if final:
        with timed_step("Loading full dataset"):
            train_texts, y_train, _, _, eval_texts, y_eval = load_yelp_data(
                train_size=None,
                val_split=0,
            )
        train_texts, val_texts, y_train, y_val = train_test_split(
            train_texts,
            y_train,
            test_size=0.05,
            stratify=y_train,
            random_state=0,
        )
        eval_label = "TEST"
    else:
        with timed_step("Loading dataset"):
            train_texts, y_train, val_texts, y_val, _, _ = load_yelp_data(
                train_size=50000,
                val_split=0.1,
            )
        eval_texts = val_texts
        y_eval = y_val
        eval_label = "validation"

    assert val_texts is not None and y_val is not None
    assert eval_texts is not None and y_eval is not None

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_eval = np.array(y_eval)

    print(f"Train: {len(train_texts)} | {eval_label}: {len(eval_texts)} | Device: {DEVICE}")

    with timed_step("Building vocabulary"):
        vocab = build_vocab(train_texts, max_vocab=params["max_vocab"])
    with timed_step("Tokenizing texts"):
        X_train = texts_to_indices(train_texts, vocab, max_len=params["max_len"])
        X_val = texts_to_indices(val_texts, vocab, max_len=params["max_len"])
        X_eval = texts_to_indices(eval_texts, vocab, max_len=params["max_len"])

    with timed_step("Training model"):
        model = build_model(len(vocab), params)
        model = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )

    with timed_step("Evaluating model"):
        y_pred, metrics = evaluate(
            model,
            X_eval,
            y_eval,
            batch_size=params["batch_size"],
            model_name=model_name,
        )
    cm_path = os.path.join(SCRIPT_DIR, "confusion_matrix_transformer.png")
    plot_confusion_matrix(y_eval, y_pred, cm_path, model_name)

    total = time.time() - run_start
    save_results(model_name, metrics, total, RESULTS_LOG, final=final)
    print(f"Done in {total:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--final", action="store_true", help="Full training set, evaluate on test")
    args = parser.parse_args()

    if args.tune:
        run_tuning()
    else:
        main(final=args.final)
