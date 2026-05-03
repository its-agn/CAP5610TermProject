import numpy as np
import torch
import torch.nn as nn
import os, sys, time, re
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# !IMPORTANT! if not good, increase max_len in revs to indices, or vocab size

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils import (
    DEFAULT_SEED,
    common_parser,
    compute_metrics,
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TUNING_LOG = os.path.join(SCRIPT_DIR, "tuning_log.md")
RESULTS_LOG = os.path.join(SCRIPT_DIR, "results_log.md")
BEST_PARAMS_FILE = os.path.join(SCRIPT_DIR, "best_params.json")
DEFAULT_TRAIN_SIZE = 350_000
FINAL_TRAIN_SIZE = 650_000
TUNING_TRAIN_SIZE = 50_000

parser = common_parser()
args = parser.parse_args()

# * Preprocessing Utils

punctuation_rem = re.compile((r"[^\w\s]"))
def _clean_text(text):
    return punctuation_rem.sub(' ', text.lower())

def build_vocab(revs, max_vocab = 50000):
    counter = Counter()
    
    for rev in revs:
        counter.update(_clean_text(rev).split())
        
    vocab = {"<pad>": 0, "<unk>": 1} # pad at 0, reserve unkowns for 1
    
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    
    print("Vocab Size: " + str(len(vocab)) + "\n")
    
    return vocab

def texts_to_indices(texts, vocab, max_len=256):
    indices = np.zeros((len(texts), max_len), dtype=np.int64)

    for i, text in enumerate(texts):
        words = _clean_text(text).split()[:max_len]

        for j, word in enumerate(words):
            indices[i, j] = vocab.get(word, 1)

    return indices


# * Model Definition

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_layers=1,
        num_classes=5,
        dropout=0.3,
        bidirectional=True
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        self.bidirectional = bidirectional
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        
        if self.bidirectional:
            pooled = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            pooled = hidden[-1]
            
        pooled = self.dropout(pooled)
        return self.fc(pooled)
    
# ! Need to optimize macro f1 thru eval & compute_metrics
    
def train_model(model, train_loader, val_data=None, epochs=5, lr=1e-3, patience=3):
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = epochs
    epochs_no_improve = 0

    if val_data is not None:
        X_val_t, y_val_arr = val_data
        y_val_t = torch.from_numpy(y_val_arr)
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t),
            batch_size=512,
            shuffle=False,
        )

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        if val_data is not None:
            model.eval()

            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for val_X, val_y in val_loader:
                    val_X = val_X.to(DEVICE)
                    val_y = val_y.to(DEVICE)

                    val_logits = model(val_X)
                    loss = criterion(val_logits, val_y)

                    val_loss += loss.item() * val_X.size(0)
                    val_correct += (val_logits.argmax(dim=1) == val_y).sum().item()
                    val_total += val_y.size(0)

            val_avg_loss = val_loss / val_total
            val_acc = val_correct / val_total

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_avg_loss:.4f} - Val Acc: {val_acc:.4f}"
            )

            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        else:
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch

            
def evaluate(model, X_eval_tensor, y_eval, batch_size=512, verbose=True):
    model.eval()

    preds = []
    loader = DataLoader(TensorDataset(X_eval_tensor), batch_size=batch_size, shuffle=False)

    with timed_step("Running Inference"):
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(DEVICE)
                logits = model(batch_X)
                preds.append(logits.argmax(dim=1).cpu().numpy())

        y_pred = np.concatenate(preds)
        metrics = compute_metrics(y_eval, y_pred)

        if verbose:
            print_metrics(metrics, "LSTM", y_eval, y_pred)

        return y_pred, metrics


def run_tuning(discard=False):
    set_random_seed(DEFAULT_SEED)
    model_name = "LSTM"
    tuning_val_split = 0.1
    tuning_trials = 15
    
    print_run_header(
        model_name,
        mode="tuning",
        device=get_device_name(),
        seed=DEFAULT_SEED,
        extra_info={
            "Dataset": f"{TUNING_TRAIN_SIZE} train ({int(tuning_val_split * 100)}% val split)",
            "Trials": tuning_trials
            }
    )
    
    with timed_step("Loading dataset"):
        train_texts, y_train, val_texts, y_val, _, _ = load_yelp_data(
            train_size=TUNING_TRAIN_SIZE,
            val_split=tuning_val_split,
            skip_test=True,
            seed=DEFAULT_SEED,
        )
        
    assert val_texts is not None and y_val is not None
    
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    def objective(trial):
        set_random_seed(DEFAULT_SEED)

        params = {
            "max_vocab": trial.suggest_categorical("max_vocab", [10000, 20000, 30000, 40000, 50000, 75000]),
            "max_len": trial.suggest_categorical("max_len", [128, 256, 512]),
            "embed_dim": trial.suggest_categorical("embed_dim", [128, 256, 300]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256]),
            "num_layers": trial.suggest_categorical("num_layers", [1, 2]),
            "bidirectional": trial.suggest_categorical("bidirectional", [True]),
            "dropout": trial.suggest_float("dropout", 0.2, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "epochs": trial.suggest_int("epochs", 3, 15),
            "patience": 3,
        }

        with timed_step("  Building vocabulary"):
            vocab = build_vocab(train_texts, max_vocab=params["max_vocab"])

        with timed_step("  Tokenizing texts"):
            X_train = texts_to_indices(train_texts, vocab, max_len=params["max_len"])
            X_val = texts_to_indices(val_texts, vocab, max_len=params["max_len"])

        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
        )

        model = LSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=params["embed_dim"],
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            bidirectional=params["bidirectional"],
            dropout=params["dropout"],
        )

        model, best_epoch = train_model(
            model,
            train_loader,
            val_data=(torch.from_numpy(X_val), y_val),
            epochs=params["epochs"],
            lr=params["lr"],
            patience=params["patience"],
        )

        trial.set_user_attr("best_epoch", best_epoch)

        _, metrics = evaluate(
            model,
            torch.from_numpy(X_val),
            y_val,
            batch_size=params["batch_size"],
            verbose=False,
        )

        return metrics["macro_f1"]

    results = tune_model(
        objective,
        n_trials=tuning_trials,
        log_path=None if discard else TUNING_LOG,
        model_name=model_name,
        extra_info={
            "Dataset": f"{TUNING_TRAIN_SIZE} train ({int(tuning_val_split * 100)}% val split)",
            "Trials": tuning_trials,
        },
        seed=DEFAULT_SEED,
    )

    if not discard:
        best_config = dict(results["best_config"])
        best_epoch = results["best_user_attrs"].get("best_epoch")

        if best_epoch is not None:
            best_config["epochs"] = best_epoch

        save_best_config(
            best_config,
            BEST_PARAMS_FILE,
            metadata={
                "seed": DEFAULT_SEED,
                "tuning_train_size": len(train_texts) + len(val_texts),
                "tuning_val_split": tuning_val_split,
                "tuning_trials": tuning_trials,
            },
            macro_f1=results["best_score"],
        )
    

    
def main(final=False, discard=False, default_config=False):
    set_random_seed(DEFAULT_SEED)
    run_start = time.monotonic()
    model_name = "LSTM"

    default_params = {
        "max_vocab": 50000,
        "max_len": 256,
        "embed_dim": 128,
        "hidden_dim": 128,
        "num_layers": 1,
        "bidirectional": True,
        "dropout": 0.3,
        "lr": 1e-3,
        "batch_size": 128,
        "epochs": 5,
        "patience": 3,
    }

    params, metadata = (None, {}) if default_config else load_best_config(BEST_PARAMS_FILE)
    using_defaults = params is None

    if params is None:
        params = dict(default_params)
    else:
        params = {**default_params, **params}

    params_source = "default config" if using_defaults else "best tuned config"

    print_run_header(
        model_name,
        mode="final" if final else "validation",
        device=get_device_name(),
        seed=DEFAULT_SEED,
        extra_info={"Config source": params_source},
    )
    print_value_section("Parameters", params)

    if final:
        with timed_step("Loading full dataset (650k, no subsampling)"):
            train_texts, y_train, _, _, eval_texts, y_eval = load_yelp_data(
                train_size=None,
                val_split=0,
                seed=DEFAULT_SEED,
            )
        eval_label = "TEST"
    else:
        with timed_step("Loading dataset (subsampled to 150k)"):
            train_texts, y_train, eval_texts, y_eval, _, _ = load_yelp_data(
                train_size=150000,
                val_split=0.1,
                skip_test=True,
                seed=DEFAULT_SEED,
            )
        eval_label = "validation"

    assert eval_texts is not None and y_eval is not None

    y_train = np.array(y_train)
    y_eval = np.array(y_eval)

    print(f"Train: {len(train_texts)} | {eval_label}: {len(eval_texts)}")

    with timed_step("Building vocabulary"):
        vocab = build_vocab(train_texts, max_vocab=params["max_vocab"])

    with timed_step("Tokenizing texts"):
        X_train = texts_to_indices(train_texts, vocab, max_len=params["max_len"])
        X_eval = texts_to_indices(eval_texts, vocab, max_len=params["max_len"])

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
    )

    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=params["embed_dim"],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        bidirectional=params["bidirectional"],
        dropout=params["dropout"],
    )

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    val_data = None if final else (torch.from_numpy(X_eval), y_eval)

    with timed_step("Training model"):
        model, best_epoch = train_model(
            model,
            train_loader,
            val_data=val_data,
            epochs=params["epochs"],
            lr=params["lr"],
            patience=params["patience"],
        )

    X_eval_tensor = torch.from_numpy(X_eval)

    with timed_step("Evaluating model"):
        y_pred, metrics = evaluate(
            model,
            X_eval_tensor,
            y_eval,
            batch_size=params["batch_size"],
            verbose=True,
        )

    total = time.monotonic() - run_start

    if not discard:
        saved = save_results(
            model_name,
            metrics,
            total,
            RESULTS_LOG,
            final=final,
            device=get_device_name(),
            default_config=using_defaults,
            params=params,
            metadata={
                "seed": DEFAULT_SEED,
                "best_epoch": best_epoch,
                "vocab_size": len(vocab),
            } if using_defaults else metadata,
        )

        if final and saved:
            cm_path = os.path.join(SCRIPT_DIR, "confusion_matrix_lstm.png")
            plot_confusion_matrix(
                y_eval,
                y_pred,
                cm_path,
                model_name,
                title_suffix=f"Macro F1={metrics['macro_f1']:.4f}",
            )
    else:
        print("Skipping save (--discard)")

    print(f"\nTotal time: {total:.1f}s ({total / 60:.1f}m)")
    
if __name__ == "__main__":
    if args.tune:
        if args.final or args.default:
            parser.error("tune can't be run with final or default")
        run_tuning(discard=args.discard)
    else: 
        main(final=args.final, discard=args.discard, default_config=args.default)
