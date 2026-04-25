import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import scipy.sparse as sp

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    load_yelp_data,
    fit_tfidf_features,
    compute_metrics,
    print_metrics,
    print_run_header,
    plot_confusion_matrix,
    save_results,
    get_device_name,
    timed_step,
    set_random_seed,
    load_best_config,
    save_best_config,
    tune_model,
    write_tuning_log,
    LABEL_NAMES,
    common_parser,
    DEFAULT_SEED,
)

SEED        = DEFAULT_SEED
MODEL_NAME  = "MLP"
BEST_CFG    = "mlp_best_config.json"
TUNE_LOG    = "mlp_tuning_log.md"
RESULTS_LOG = "results_log.md"
CM_PATH     = "mlp_confusion_matrix.png"
TFIDF_DIM   = 10000
BATCH_SIZE  = 256

set_random_seed(SEED)

args   = common_parser().parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print_run_header(
    MODEL_NAME,
    device=get_device_name(),
    seed=SEED,
    extra_info={"TF-IDF features": TFIDF_DIM, "Batch size": BATCH_SIZE},
)


# data loading
train_size = None if args.final else 150000
train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_yelp_data(
    train_size=train_size, val_split=0.1, skip_test=not args.final
)

eval_texts  = test_texts  if args.final else val_texts
eval_labels = test_labels if args.final else val_labels

X_train_sp, X_eval_sp = fit_tfidf_features(
    train_texts, eval_texts,
    max_features=TFIDF_DIM,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
)


def sparse_to_loader(X_sp, y, batch_size, shuffle, chunk_size=10000):
    """Convert sparse TF-IDF matrix to a DataLoader without blowing up RAM.
    Converts in chunks to avoid the full dense allocation."""
    chunks = []
    n = X_sp.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = torch.tensor(X_sp[start:end].toarray(), dtype=torch.float32)
        chunks.append(chunk)
    X_dense = torch.cat(chunks, dim=0)
    y_tensor = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(X_dense, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=(device.type == "cuda"))


with timed_step("Building train DataLoader"):
    train_loader = sparse_to_loader(X_train_sp, train_labels, BATCH_SIZE, shuffle=True)

with timed_step("Building eval DataLoader"):
    eval_loader  = sparse_to_loader(X_eval_sp,  eval_labels,  BATCH_SIZE, shuffle=False)

INPUT_DIM   = TFIDF_DIM
NUM_CLASSES = 5


class MLP(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        preds          = logits.argmax(dim=1)
        total_correct += (preds == y_batch).sum().item()
        total_samples += y_batch.size(0)
        total_loss    += loss.item() * y_batch.size(0)

    return total_loss / total_samples, total_correct / total_samples


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            preds  = logits.argmax(dim=1)

            total_correct += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)
            total_loss    += loss.item() * y_batch.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return (total_loss / total_samples,
            total_correct / total_samples,
            np.array(all_preds),
            np.array(all_labels))


def run_training(config, tag="run", max_epochs=10, patience=3):
    """Train model with given config. Early stopping on val accuracy."""
    model = MLP(
        input_dim=INPUT_DIM,
        hidden1=config["hidden1"],
        hidden2=config["hidden2"],
        num_classes=NUM_CLASSES,
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    train_accs, val_accs     = [], []
    train_losses, val_losses = [], []
    best_val_acc   = 0.0
    best_state     = None
    patience_count = 0

    print(f"\n[{tag}] starting -- max {max_epochs} epochs, patience {patience}")
    print("-" * 52)

    import time
    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        t_ep = time.time()
        tr_loss, tr_acc         = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, _, _   = eval_epoch(model, eval_loader,  criterion)

        scheduler.step(vl_acc)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        marker = ""
        if vl_acc > best_val_acc:
            best_val_acc   = vl_acc
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
            marker = " *"
        else:
            patience_count += 1

        print(f"  epoch {epoch:02d}/{max_epochs} | "
              f"train acc: {tr_acc:.4f}  loss: {tr_loss:.4f} | "
              f"val acc: {vl_acc:.4f}  loss: {vl_loss:.4f} | "
              f"{time.time()-t_ep:.1f}s{marker}")

        if patience_count >= patience:
            print(f"  early stopping at epoch {epoch}")
            break

    total_time = time.time() - t_start
    print(f"\n[{tag}] done -- {total_time:.1f}s | best val acc: {best_val_acc:.4f}")

    model.load_state_dict(best_state)
    return model, {
        "train_acc": train_accs, "val_acc": val_accs,
        "train_loss": train_losses, "val_loss": val_losses,
        "total_time": total_time, "best_val_acc": best_val_acc,
    }


# default config run
default_config = {
    "hidden1":      1024,
    "hidden2":      512,
    "dropout":      0.5,
    "lr":           1e-3,
    "weight_decay": 1e-4,
}

print("\n=== default config run ===")
default_model, default_history = run_training(default_config, tag="default")


# hyperparameter tuning via Optuna if --tune or no saved config
saved_params, saved_meta = load_best_config(BEST_CFG)
run_tune = args.tune or (saved_params is None and not args.default)

if run_tune:
    print("\n=== optuna hyperparameter tuning ===")

    criterion = nn.CrossEntropyLoss()

    def objective(trial):
        cfg = {
            "hidden1":      trial.suggest_categorical("hidden1", [512, 1024, 2048]),
            "hidden2":      trial.suggest_categorical("hidden2", [256, 512]),
            "dropout":      trial.suggest_float("dropout", 0.1, 0.6, step=0.1),
            "lr":           trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        }
        # short runs during tuning -- 5 epochs, patience 2
        _, history = run_training(cfg, tag=f"trial", max_epochs=5, patience=2)
        return history["best_val_acc"]

    results = tune_model(
        objective,
        n_trials=20,
        log_path=None if args.discard else TUNE_LOG,
        model_name=MODEL_NAME,
        seed=SEED,
    )

    best_config = results["best_config"]

    if not args.discard:
        save_best_config(
            best_config, BEST_CFG,
            metadata={"tfidf_dim": TFIDF_DIM, "ngram_range": "(1,2)"},
            macro_f1=results["best_score"],
        )

elif saved_params and not args.default:
    best_config = saved_params
    print(f"\nusing saved best config from {BEST_CFG}")
else:
    best_config = default_config
    print("\nusing default config for final run")


# final tuned run with full epochs
print("\n=== final tuned model ===")
tuned_model, tuned_history = run_training(best_config, tag="tuned", max_epochs=15, patience=4)


# evaluate both models
criterion = nn.CrossEntropyLoss()

_, _, default_preds, default_labels = eval_epoch(default_model, eval_loader, criterion)
_, _, tuned_preds,   tuned_labels   = eval_epoch(tuned_model,   eval_loader, criterion)

split = "test" if args.final else "val"

print(f"\n--- default ({split}) ---")
default_metrics = compute_metrics(default_labels, default_preds)
print_metrics(default_metrics, f"{MODEL_NAME} (default)", y_true=default_labels, y_pred=default_preds)

print(f"\n--- tuned ({split}) ---")
tuned_metrics = compute_metrics(tuned_labels, tuned_preds)
print_metrics(tuned_metrics, f"{MODEL_NAME} (tuned)", y_true=tuned_labels, y_pred=tuned_preds)


# save results
if not args.discard:
    save_results(
        MODEL_NAME, default_metrics, default_history["total_time"],
        RESULTS_LOG, final=args.final, device=get_device_name(),
        default_config=True, params=default_config,
    )
    save_results(
        MODEL_NAME, tuned_metrics, tuned_history["total_time"],
        RESULTS_LOG, final=args.final, device=get_device_name(),
        default_config=False, params=best_config,
    )

# confusion matrix for tuned model
plot_confusion_matrix(
    tuned_labels, tuned_preds,
    save_path=CM_PATH,
    model_name=MODEL_NAME,
    title_suffix=f"tuned | {split} set",
)


# plots: accuracy and loss curves side by side
def plot_history(history_a, label_a, history_b, label_b, metric, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, hist, label in zip(axes, [history_a, history_b], [label_a, label_b]):
        x = range(1, len(hist[f"train_{metric}"]) + 1)
        ax.plot(x, hist[f"train_{metric}"], marker="o", label=f"train {metric}", color="steelblue")
        ax.plot(x, hist[f"val_{metric}"],   marker="o", label=f"val {metric}",   color="orange")
        ax.set_title(f"MLP ({label}) -- {metric.title()} over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"saved: {outpath}")

plot_history(default_history, "default", tuned_history, "tuned", "acc",  "mlp_accuracy.png")
plot_history(default_history, "default", tuned_history, "tuned", "loss", "mlp_loss.png")