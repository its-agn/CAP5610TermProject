"""
CNN (TextCNN) on Yelp Review Full (5-class star rating prediction).
Keating Sane - CAP5610 Spring 2026

Usage: python CNN.py [common flags] [--glove-6b | --no-glove]
"""
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    build_embedding_matrix,
    common_parser,
    compute_metrics,
    load_best_params,
    load_yelp_data,
    plot_confusion_matrix,
    print_metrics,
    save_best_params,
    save_results,
    timed_step,
    tune_model,
)

GLOVE_SOURCE = "42B"
GLOVE_DIM = 300

if __name__ == "__main__":
    parser = common_parser()
    parser.add_argument("--no-glove", action="store_true",
                        help="Use random embeddings instead of GloVe")
    parser.add_argument("--glove-6b", action="store_true",
                        help="Use GloVe 6B 100d instead of 42B 300d")
    args = parser.parse_args()
    if args.tune and (args.final or args.default_params or args.no_glove):
        parser.error("--tune cannot be combined with --final, --default-params, or --no-glove")
    if args.no_glove and args.glove_6b:
        parser.error("--no-glove and --glove-6b cannot be combined")
    if args.glove_6b:
        GLOVE_SOURCE = "6B"
        GLOVE_DIM = 100

with timed_step("Loading libraries"):
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

import logging
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.CRITICAL)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TUNING_LOG = os.path.join(SCRIPT_DIR, "tuning_log.md")
RESULTS_LOG = os.path.join(SCRIPT_DIR, "results_log.md")
BEST_PARAMS_FILE = os.path.join(SCRIPT_DIR, "best_params.json")

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

_PUNCT_RE = re.compile(r'[^\w\s]')

def _clean_text(text):
    """Replace punctuation with spaces to improve GloVe coverage."""
    return _PUNCT_RE.sub(' ', text.lower())

def build_vocab(texts, max_vocab=25000):
    """Build word -> index mapping from training texts."""
    counter = Counter()
    for text in texts:
        counter.update(_clean_text(text).split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab

def texts_to_indices(texts, vocab, max_len=256):
    """Convert texts to padded/truncated index sequences."""
    indices = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        words = _clean_text(text).split()[:max_len]
        for j, word in enumerate(words):
            indices[i, j] = vocab.get(word, 1)
    return indices

class TextCNN(nn.Module):
    """Kim (2014) style TextCNN: embedding -> parallel conv filters -> max-pool -> FC."""

    def __init__(self, vocab_size, embed_dim=300, num_filters=100,
                 filter_sizes=(3, 4, 5), num_classes=5, dropout=0.5,
                 pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)          # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)          # (batch, embed_dim, seq_len)
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(x))    # (batch, num_filters, seq_len - fs + 1)
            c = c.max(dim=2).values    # (batch, num_filters)
            pooled.append(c)
        x = torch.cat(pooled, dim=1)   # (batch, num_filters * len(filter_sizes))
        x = self.dropout(x)
        return self.fc(x)

def train_model(model, train_loader, epochs=5, lr=1e-3,
                val_data=None, patience=3, grad_clip=1.0, indent=""):
    with timed_step(f"{indent}Initializing model on {DEVICE}"):
        model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    if val_data is not None:
        X_val_t, y_val_arr = val_data
        y_val_t = torch.from_numpy(y_val_arr)
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t), batch_size=512, shuffle=False,
        )

    for epoch in range(epochs):
        model.train()
        epoch_start = time.monotonic()
        last_print = epoch_start
        total_loss = 0
        correct = 0
        total = 0
        n_batches = len(train_loader)

        for batch_i, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0)
            correct += (out.argmax(dim=1) == batch_y).sum().item()
            total += batch_y.size(0)
            now = time.monotonic()
            if now - last_print >= 0.1 or batch_i + 1 == n_batches:
                last_print = now
                print(f"\r{indent}Epoch {epoch+1}/{epochs} - "
                      f"Batch {batch_i+1}/{n_batches} - ({now - epoch_start:.1f}s)", end="")

        elapsed = time.monotonic() - epoch_start
        avg_loss = total_loss / total
        acc = correct / total

        if val_data is not None:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for vb_X, vb_y in val_loader:
                    vb_X, vb_y = vb_X.to(DEVICE), vb_y.to(DEVICE)
                    vout = model(vb_X)
                    vloss = criterion(vout, vb_y)
                    val_loss += vloss.item() * vb_X.size(0)
                    val_correct += (vout.argmax(dim=1) == vb_y).sum().item()
                    val_total += vb_y.size(0)
            val_avg = val_loss / val_total
            val_acc = val_correct / val_total
            print(f"\r{indent}Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - "
                  f"Train Acc: {acc:.4f} - Val Loss: {val_avg:.4f} - "
                  f"Val Acc: {val_acc:.4f} - ({elapsed:.1f}s)   ")

            if val_avg < best_val_loss:
                best_val_loss = val_avg
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"{indent}Early stopping at epoch {epoch+1} "
                          f"(no val improvement for {patience} epochs)")
                    break
        else:
            print(f"\r{indent}Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - "
                  f"Train Acc: {acc:.4f} - ({elapsed:.1f}s)   ")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate(model, X_eval_tensor, y_eval, verbose=True, model_name="TextCNN"):
    """Run inference and compute evaluation metrics."""
    model.eval()
    all_preds = []
    loader = DataLoader(TensorDataset(X_eval_tensor), batch_size=512, shuffle=False)
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(DEVICE)
            out = model(batch_X)
            all_preds.append(out.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(all_preds)

    metrics = compute_metrics(y_eval, y_pred)
    if verbose:
        print_metrics(metrics, model_name, y_eval, y_pred)
    return y_pred, metrics

def run_tuning(no_save=False):
    """Tune CNN with Optuna (Bayesian optimization)."""
    with timed_step("Loading dataset"):
        all_texts, all_labels, _, _, _, _ = load_yelp_data(train_size=None, val_split=0, skip_test=True)

    # subsample + split once (consistent across trials)
    train_texts, _, train_labels, _ = train_test_split(
        all_texts, list(all_labels),
        train_size=150000, stratify=all_labels, random_state=0,
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels,
        test_size=0.1, stratify=train_labels, random_state=0,
    )
    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    with timed_step("Building vocabulary"):
        vocab = build_vocab(train_texts)

    with timed_step("Tokenizing texts"):
        X_train = texts_to_indices(train_texts, vocab)
        X_val = texts_to_indices(val_texts, vocab)

    embed_matrix = build_embedding_matrix(vocab, source=GLOVE_SOURCE, dim=GLOVE_DIM)
    X_val_tensor = torch.from_numpy(X_val)

    def objective(trial):
        num_filters = trial.suggest_categorical("num_filters", [100, 200, 300])
        filter_cfg = trial.suggest_categorical("filter_sizes", ["3,4,5", "2,3,4,5"])
        filter_sizes = tuple(int(x) for x in filter_cfg.split(","))
        dropout = trial.suggest_float("dropout", 0.2, 0.6)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        epochs = trial.suggest_int("epochs", 5, 15)

        train_dataset = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
        )

        model = TextCNN(
            vocab_size=len(vocab),
            embed_dim=GLOVE_DIM,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            dropout=dropout,
            pretrained_embeddings=embed_matrix,
        )

        model = train_model(model, train_loader, epochs=epochs, lr=lr,
                            val_data=(X_val_tensor, y_val), indent="  ")

        _, metrics = evaluate(model, X_val_tensor, y_val, verbose=False)
        return metrics["macro_f1"]

    results = tune_model(
        objective,
        n_trials=30,
        log_path=None if no_save else TUNING_LOG,
        best_params_path=None,
        model_name="TextCNN",
    )
    if not no_save:
        params = results["best_params"]
        params["glove_source"] = GLOVE_SOURCE
        params["glove_dim"] = GLOVE_DIM
        save_best_params(params, BEST_PARAMS_FILE, score=results["best_score"])

def main(final=False, use_glove=True, no_save=False, default_params=False):
    run_start = time.monotonic()
    if not use_glove:
        model_name = "TextCNN (no GloVe)"
    else:
        model_name = f"TextCNN (GloVe {GLOVE_SOURCE} {GLOVE_DIM}d)"

    DEFAULT_PARAMS = {
        "num_filters": 100, "filter_sizes": "3,4,5", "dropout": 0.5,
        "lr": 1e-3, "batch_size": 64, "epochs": 10,
    }

    params = None if default_params else load_best_params(BEST_PARAMS_FILE)
    using_defaults = params is None
    if not using_defaults:
        saved_source = params.pop("glove_source", None)
        saved_dim = params.pop("glove_dim", None)
        if saved_source and use_glove and (saved_source != GLOVE_SOURCE or saved_dim != GLOVE_DIM):
            sys.exit(f"Error: best params were tuned with GloVe {saved_source} {saved_dim}d "
                     f"but running with GloVe {GLOVE_SOURCE} {GLOVE_DIM}d. "
                     f"Use --default-params to ignore best params.")
        elif saved_source and not use_glove:
            sys.exit(f"Error: best params were tuned with GloVe {saved_source} {saved_dim}d "
                     f"but running without GloVe (--no-glove). "
                     f"Use --default-params to ignore best params.")
    if using_defaults:
        params = DEFAULT_PARAMS
        print("Using default params:")
    else:
        print("Using best tuned params (from best_params.json):")
    for k, v in params.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if final:
        with timed_step("Loading full dataset (650k, no subsampling)"):
            train_texts, y_train, _, _, eval_texts, y_eval = load_yelp_data(
                train_size=None, val_split=0,
            )
        eval_label = "TEST"
    else:
        with timed_step("Loading dataset (subsampled to 150k)"):
            train_texts, y_train, eval_texts, y_eval, _, _ = load_yelp_data(
                train_size=150000,
            )
        eval_label = "validation"

    assert eval_texts is not None and y_eval is not None
    print(f"Train: {len(train_texts)} | {eval_label}: {len(eval_texts)}")

    with timed_step("Building vocabulary"):
        vocab = build_vocab(train_texts)

    with timed_step("Tokenizing texts"):
        X_train = texts_to_indices(train_texts, vocab)
        X_eval = texts_to_indices(eval_texts, vocab)

    embed_matrix = build_embedding_matrix(vocab, source=GLOVE_SOURCE, dim=GLOVE_DIM) if use_glove else None
    embed_dim = GLOVE_DIM if use_glove else 256

    fs = params["filter_sizes"]
    filter_sizes = tuple(int(x) for x in fs.split(",")) if isinstance(fs, str) else fs
    num_filters = params["num_filters"]
    dropout = params["dropout"]
    lr = params["lr"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]

    # Early stopping validation split (avoid using test set in --final mode)
    if final:
        n_val = int(len(X_train) * 0.05)
        perm = np.random.RandomState(42).permutation(len(X_train))
        es_X = torch.from_numpy(X_train[perm[:n_val]])
        es_y = y_train[perm[:n_val]]
        X_train = X_train[perm[n_val:]]
        y_train = y_train[perm[n_val:]]
        val_data = (es_X, es_y)
    else:
        val_data = (torch.from_numpy(X_eval), y_eval)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_filters=num_filters,
        filter_sizes=filter_sizes,
        dropout=dropout,
        pretrained_embeddings=embed_matrix,
    )
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    model = train_model(model, train_loader, epochs=epochs, lr=lr,
                        val_data=val_data)

    X_eval_tensor = torch.from_numpy(X_eval)
    y_pred, metrics = evaluate(model, X_eval_tensor, y_eval, model_name=model_name)

    total = time.monotonic() - run_start
    if not no_save:
        cm_path = os.path.join(SCRIPT_DIR, "confusion_matrix_cnn.png")
        plot_confusion_matrix(y_eval, y_pred, cm_path, model_name)
        save_results(model_name, metrics, total, RESULTS_LOG, final=final,
                     default_params=using_defaults)
    else:
        print("Skipping save (--no-save)")
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f}m)")

if __name__ == "__main__":
    if args.tune:
        run_tuning(no_save=args.no_save)
    else:
        main(final=args.final, use_glove=not args.no_glove, no_save=args.no_save,
             default_params=args.default_params)
