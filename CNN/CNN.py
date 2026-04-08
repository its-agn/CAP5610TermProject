"""
CNN (TextCNN) on Yelp Review Full (5-class star rating prediction).
Keating Sane - CAP5610 Spring 2026

Usage: python CNN.py [common flags] [--glove-6b-300d | --glove-6b-100d | --glove-42b | --glove-2024-wikigiga | --fasttext-wiki-subword]
"""
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    DEFAULT_SEED,
    build_embedding_matrix,
    common_parser,
    compute_metrics,
    embedding_display_name,
    get_device_name,
    load_best_config,
    load_yelp_data,
    plot_confusion_matrix,
    print_metrics,
    print_run_header,
    print_value_section,
    save_best_config,
    save_results,
    set_random_seed,
    timed_step,
    tune_model,
)

EMBEDDING_SOURCE = "6B"
EMBEDDING_DIM = 300
USE_PRETRAINED_EMBEDDINGS = False
TUNING_TRAIN_SIZE = 650000
TUNING_VAL_SPLIT = 0.1
TUNING_TRIALS = 30

def parse_args():
    """Parse CLI args and configure embedding globals."""
    global USE_PRETRAINED_EMBEDDINGS, EMBEDDING_SOURCE, EMBEDDING_DIM
    parser = common_parser()
    parser.add_argument("--glove-6b-300d", action="store_true",
                        help="Use GloVe 6B 300d instead of the default random embeddings")
    parser.add_argument("--glove-42b", action="store_true",
                        help="Use GloVe 42B 300d instead of the default random embeddings")
    parser.add_argument("--glove-6b-100d", action="store_true",
                        help="Use GloVe 6B 100d instead of the default random embeddings")
    parser.add_argument("--glove-2024-wikigiga", action="store_true",
                        help="Use GloVe 2024 WikiGigaword 300d instead of the default random embeddings")
    parser.add_argument("--fasttext-wiki-subword", action="store_true",
                        help="Use fastText wiki-news subword 300d instead of the default random embeddings")
    args = parser.parse_args()
    if args.tune and (args.final or args.default):
        parser.error("--tune cannot be combined with --final or --default")
    if sum(bool(x) for x in (
        args.glove_6b_300d,
        args.glove_42b,
        args.glove_6b_100d,
        args.glove_2024_wikigiga,
        args.fasttext_wiki_subword,
    )) > 1:
        parser.error(
            "--glove-6b-300d, --glove-42b, --glove-6b-100d, "
            "--glove-2024-wikigiga, and --fasttext-wiki-subword are mutually exclusive"
        )
    if args.glove_6b_300d:
        USE_PRETRAINED_EMBEDDINGS = True
        EMBEDDING_SOURCE = "6B"
        EMBEDDING_DIM = 300
    elif args.glove_42b:
        USE_PRETRAINED_EMBEDDINGS = True
        EMBEDDING_SOURCE = "42B"
        EMBEDDING_DIM = 300
    elif args.glove_6b_100d:
        USE_PRETRAINED_EMBEDDINGS = True
        EMBEDDING_SOURCE = "6B"
        EMBEDDING_DIM = 100
    elif args.glove_2024_wikigiga:
        USE_PRETRAINED_EMBEDDINGS = True
        EMBEDDING_SOURCE = "2024WG"
        EMBEDDING_DIM = 300
    elif args.fasttext_wiki_subword:
        USE_PRETRAINED_EMBEDDINGS = True
        EMBEDDING_SOURCE = "FT-WIKI-SUBWORD"
        EMBEDDING_DIM = 300
    return args

if __name__ == "__main__":
    args = parse_args()

with timed_step("Loading libraries"):
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TUNING_LOG = os.path.join(SCRIPT_DIR, "tuning_log.md")
RESULTS_LOG = os.path.join(SCRIPT_DIR, "results_log.md")
BEST_CONFIG_FILE = os.path.join(SCRIPT_DIR, "best_config.json")

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

def current_embedding_name():
    """Return the current embedding display name."""
    return embedding_display_name(EMBEDDING_SOURCE, EMBEDDING_DIM)

def current_embedding_label():
    """Return the current embedding label or Random when none is selected."""
    return current_embedding_name() if USE_PRETRAINED_EMBEDDINGS else "Random"

def current_embedding_results_label():
    """Return the result-log label for the active embedding setup."""
    return current_embedding_name() if USE_PRETRAINED_EMBEDDINGS else "Random embeddings"

def current_embedding_metadata():
    """Return metadata for the currently selected embedding setup."""
    return {
        "embedding_source": EMBEDDING_SOURCE if USE_PRETRAINED_EMBEDDINGS else "random",
        "embedding_dim": EMBEDDING_DIM if USE_PRETRAINED_EMBEDDINGS else 256,
        "seed": DEFAULT_SEED,
    }

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
    best_epoch = epochs
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
                best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if epoch + 1 < epochs:
                        print(f"{indent}Early stopping at epoch {epoch+1} "
                              f"(no val improvement for {patience} epochs)")
                    break
        else:
            print(f"\r{indent}Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - "
                  f"Train Acc: {acc:.4f} - ({elapsed:.1f}s)   ")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch

def evaluate(model, X_eval_tensor, y_eval, verbose=True, model_name="TextCNN", indent=""):
    """Run inference and compute evaluation metrics."""
    model.eval()
    all_preds = []
    loader = DataLoader(TensorDataset(X_eval_tensor), batch_size=512, shuffle=False)
    with timed_step(f"{indent}Running inference"):
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

def run_tuning(discard=False):
    """Tune CNN with Optuna (Bayesian optimization)."""
    set_random_seed(DEFAULT_SEED)
    print_run_header(
        "TextCNN",
        mode="tuning",
        device=get_device_name(),
        extra_info={
            "Embeddings": current_embedding_label(),
            "Dataset": f"{TUNING_TRAIN_SIZE} train ({int(TUNING_VAL_SPLIT * 100)}% val split)",
            "Trials": TUNING_TRIALS,
        },
    )

    with timed_step("Loading dataset"):
        train_texts, y_train, val_texts, y_val, _, _ = load_yelp_data(
            train_size=TUNING_TRAIN_SIZE,
            val_split=TUNING_VAL_SPLIT,
            skip_test=True,
            seed=DEFAULT_SEED,
        )
    assert val_texts is not None and y_val is not None
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    with timed_step("Building vocabulary"):
        vocab = build_vocab(train_texts)

    with timed_step("Tokenizing texts"):
        X_train = texts_to_indices(train_texts, vocab)
        X_val = texts_to_indices(val_texts, vocab)

    embed_matrix = (
        build_embedding_matrix(vocab, source=EMBEDDING_SOURCE, dim=EMBEDDING_DIM)
        if USE_PRETRAINED_EMBEDDINGS else None
    )
    X_val_tensor = torch.from_numpy(X_val)

    def objective(trial):
        set_random_seed(DEFAULT_SEED)
        num_filters = trial.suggest_categorical("num_filters", [200, 300])
        filter_cfg = trial.suggest_categorical("filter_sizes", ["2,3,4,5"])
        filter_sizes = tuple(int(x) for x in filter_cfg.split(","))
        dropout = trial.suggest_float("dropout", 0.18, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 6e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        epochs = trial.suggest_int("epochs", 5, 10)

        train_dataset = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
        )

        model = TextCNN(
            vocab_size=len(vocab),
            embed_dim=EMBEDDING_DIM if USE_PRETRAINED_EMBEDDINGS else 256,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            dropout=dropout,
            pretrained_embeddings=embed_matrix,
        )

        model, best_epoch = train_model(
            model,
            train_loader,
            epochs=epochs,
            lr=lr,
            val_data=(X_val_tensor, y_val),
            indent="  ",
        )
        trial.set_user_attr("best_epoch", best_epoch)

        _, metrics = evaluate(model, X_val_tensor, y_val, verbose=False, indent="  ")
        return metrics["macro_f1"]

    results = tune_model(
        objective,
        n_trials=TUNING_TRIALS,
        log_path=None if discard else TUNING_LOG,
        model_name="TextCNN",
        extra_info={
            "Embeddings": current_embedding_label(),
            "Dataset": f"{TUNING_TRAIN_SIZE} train ({int(TUNING_VAL_SPLIT * 100)}% val split)",
            "Trials": TUNING_TRIALS,
        },
        seed=DEFAULT_SEED,
    )
    if not discard:
        best_config = dict(results["best_config"])
        best_epoch = results["best_user_attrs"].get("best_epoch")
        if best_epoch is not None:
            best_config["epochs"] = best_epoch
        save_best_config(
            best_config, BEST_CONFIG_FILE,
            metadata={
                "embedding_source": EMBEDDING_SOURCE if USE_PRETRAINED_EMBEDDINGS else "random",
                "embedding_dim": EMBEDDING_DIM if USE_PRETRAINED_EMBEDDINGS else 256,
                "seed": DEFAULT_SEED,
                "tuning_train_size": len(train_texts) + len(val_texts),
                "tuning_val_split": TUNING_VAL_SPLIT,
                "tuning_trials": TUNING_TRIALS,
            },
            macro_f1=results["best_score"],
        )

def main(final=False, use_glove=True, discard=False, default_config=False):
    set_random_seed(DEFAULT_SEED)
    run_start = time.monotonic()
    model_name = "TextCNN"

    DEFAULT_PARAMS = {
        "num_filters": 100, "filter_sizes": "3,4,5", "dropout": 0.5,
        "lr": 1e-3, "batch_size": 64, "epochs": 10,
    }

    params, metadata = (None, {}) if default_config else load_best_config(BEST_CONFIG_FILE)
    using_defaults = params is None
    if not using_defaults:
        saved_source = metadata.get("embedding_source")
        saved_dim = metadata.get("embedding_dim")
        if saved_source is not None and saved_dim is not None and use_glove and (
            saved_source != EMBEDDING_SOURCE or saved_dim != EMBEDDING_DIM
        ):
            sys.exit(f"Error: best config was tuned with {embedding_display_name(saved_source, saved_dim)} "
                     f"but running with {current_embedding_name()}. "
                     f"Use --default to ignore best config.")
        elif saved_source is not None and saved_dim is not None and not use_glove:
            sys.exit(f"Error: best config was tuned with {embedding_display_name(saved_source, saved_dim)} "
                     f"but running with random embeddings. "
                     f"Use --default to ignore best config.")
    if using_defaults:
        params = DEFAULT_PARAMS
        params_source = "default config"
    else:
        params_source = "best tuned config"
    print_run_header(
        model_name,
        mode="final" if final else "validation",
        device=get_device_name(),
        extra_info={
            "Embeddings": "Random" if not use_glove else current_embedding_name(),
            "Config source": params_source,
        },
    )
    print_value_section("Parameters", params)

    if final:
        with timed_step("Loading full dataset (650k, no subsampling)"):
            train_texts, y_train, _, _, eval_texts, y_eval = load_yelp_data(
                train_size=None, val_split=0, seed=DEFAULT_SEED,
            )
        eval_label = "TEST"
    else:
        with timed_step("Loading dataset (subsampled to 150k)"):
            train_texts, y_train, eval_texts, y_eval, _, _ = load_yelp_data(
                train_size=150000,
                seed=DEFAULT_SEED,
            )
        eval_label = "validation"

    assert eval_texts is not None and y_eval is not None
    print(f"Train: {len(train_texts)} | {eval_label}: {len(eval_texts)}")

    with timed_step("Building vocabulary"):
        vocab = build_vocab(train_texts)

    with timed_step("Tokenizing texts"):
        X_train = texts_to_indices(train_texts, vocab)
        X_eval = texts_to_indices(eval_texts, vocab)

    embed_matrix = build_embedding_matrix(vocab, source=EMBEDDING_SOURCE, dim=EMBEDDING_DIM) if use_glove else None
    embed_dim = EMBEDDING_DIM if use_glove else 256

    fs = params["filter_sizes"]
    filter_sizes = tuple(int(x) for x in fs.split(",")) if isinstance(fs, str) else fs
    num_filters = params["num_filters"]
    dropout = params["dropout"]
    lr = params["lr"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]

    if final:
        val_data = None
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

    model, _ = train_model(model, train_loader, epochs=epochs, lr=lr,
                           val_data=val_data)

    X_eval_tensor = torch.from_numpy(X_eval)
    y_pred, metrics = evaluate(model, X_eval_tensor, y_eval, model_name=model_name)

    total = time.monotonic() - run_start
    if not discard:
        saved = save_results(
            model_name,
            metrics,
            total,
            RESULTS_LOG,
            final=final,
            default_config=using_defaults,
            params=params,
            metadata=(current_embedding_metadata() if using_defaults else metadata),
            results_name=f"{model_name} | {current_embedding_results_label()}",
        )
        if final and not default_config and saved:
            cm_path = os.path.join(SCRIPT_DIR, "confusion_matrix_cnn.png")
            plot_confusion_matrix(
                y_eval,
                y_pred,
                cm_path,
                model_name,
                title_suffix=f"Macro F1={metrics['macro_f1']:.4f}",
            )
        elif final and not default_config:
            print("Keeping existing confusion matrix because results were not updated")
    else:
        print("Skipping save (--discard)")
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f}m)")

if __name__ == "__main__":
    if args.tune:
        run_tuning(discard=args.discard)
    else:
        main(final=args.final, use_glove=USE_PRETRAINED_EMBEDDINGS, discard=args.discard,
             default_config=args.default)
