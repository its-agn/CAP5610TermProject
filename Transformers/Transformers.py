'''
Transformers (Encoder-only) on YelpReviewFull dataset (5-class star rating prediction).
Anthony Mahon - CAP5610 Spring 2026

Run modes:
    python Transformers.py            #Trains a baseline on 50k samples
    python Transformers.py --tune     #Runs Optuna to find the best hyperparameters
    python Transformers.py --final    #Trains on full dataset, tests on locked 50k test set
'''

#While Kernel SVM looks at word frequencies (TF-IDF), a Transformer looks at the order and context of words

import argparse
import os
import re
import sys
import time
from collections import Counter

#Tell Python to look one folder up for the utils package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    compute_metrics,
    load_best_config,
    load_yelp_data,
    plot_confusion_matrix,
    print_metrics,
    save_best_config,
    save_results,
    timed_step,
    tune_model,
)

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

#Automated tuning of the hyperparameters
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_PARAMS_FILE = os.path.join(SCRIPT_DIR, "best_params.json")
TUNING_LOG = os.path.join(SCRIPT_DIR, "tuning_log.md")

RESULTS_LOG = os.path.join(SCRIPT_DIR, "results_log.md")    #Automated results logging

#Transformers training should ideally be run on a GPU
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

#CPU users, brace yourselves for a long wait, or run this on Google Colab's free T4 GPU!
if DEVICE == torch.device("cpu"):
    print("WARNING: Using CPU for training. This will be slow (like  h o u r s  slow).\n"
          " Suggestion: Stop this program and run it on a T4 GPU (e.g. Google Colab).")

TEXT_RE = re.compile(r"[^\w\s]")    # ^ = NOT, \w = word character, \s = space character. This is a filter

#Neural networks cannot read English; they only read numbers. This function first strips out punctuation and makes everything lowercase
def clean_text(text):
    #Hunt down and delete anything that isn't a letter or a space in our text (and replace it with a space and lowercase the text)
    return TEXT_RE.sub(" ", text.lower())

#And this function builds a vocab dictionary that assigns the most common words a unique ID number, like "terrible" = 42
def build_vocab(texts, max_vocab=30000):
    counter = Counter()    #Count the number of times each word appears in our training set (word frequency)
    for text in texts:
        counter.update(clean_text(text).split())

    vocab = {"<pad>": 0, "<unk>": 1}    #If a word is not in the top 30k, it becomes <unk> (unknown)

    for word, _ in counter.most_common(max_vocab - 2):    #max_vocab-2 because we don't want to account for the defaults
        vocab[word] = len(vocab)    #Enter the word into the dictionary, assigning it a new ID number. 1st iter. -> len(vocab) = 2

    return vocab

#This function executes our English-to-ID mapping. If a review is too short, add padding. If it is too long, truncate it
def texts_to_indices(texts, vocab, max_len=128):
    indices = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        words = clean_text(text).split()[:max_len]    #Split into a list, keep only up to max_len words
        for j, word in enumerate(words):
            #It's ok if the indice row is not filled; it's padded to maintain rectangular dimensions (GPU requirement)
            indices[i, j] = vocab.get(word, 1)    #1 = default "unk"nown word ID in case word is not in vocab
    return indices


#Introducing the brain of our model: the Transformer encoder
class EncoderClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len=128,
        embed_dim=128,    #The size of the embedding vector used to represent a word
        num_heads=4,    #Attention heads
        num_layers=2,
        ff_dim=256,    #Size of hidden layer Feed-forward network
        dropout=0.2,
        num_classes=5,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)    #Translate IDs to vectors
        self.position_embedding = nn.Embedding(max_len, embed_dim)    #Give the Transformer a sense of position for each word
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,    #PyTorch defaults to [sequence_len, batch_size] which is confusing. This flips it to [batch_size, sequence_len] so it makes sense
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,    #The self-attention layers of the Transformer
            enable_nested_tensor=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)    #The final layer of the model, which outputs a single number (1-5)

    def forward(self, x):
        #If a review has 50 words, x.size(1) = 50. If we are processing 64 reviews at once, .expand_as copies our [0...49] list 64 times
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)

        padding_mask = x.eq(0)    #Identify the <pad>: 0 tokens to ignore them later (using x * valid)
        x = self.token_embedding(x) + self.position_embedding(positions)    #Merge word meaning with word location
        x = self.encoder(x, src_key_padding_mask=padding_mask)    #This is the Transformer. Words update their meaning based on context
        valid = (~padding_mask).unsqueeze(-1)    #unsqueeze(-1) adds a dimension to the tensor, which is needed for 3D multiplication
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)    #Calculate the average meaning vector for each review. clamp(min=1) prevents division by zero
        pooled = self.norm(self.dropout(pooled))
        return self.classifier(pooled)    #Return the final output, which is the 1-5 star prediction scores


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
    lr=3e-3,
    weight_decay=1e-3,    #Regularization parameter
):
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    #val_loader = DataLoader(    #We build our own val Dataloader in evaluate()
    #   TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
    #  batch_size=batch_size,
    # shuffle=False,
    #)

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
            batch_X = batch_X.to(DEVICE)    #.to(DEVICE) transfers the data from RAM to the GPU to do fast math
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()    #Clear the gradient memory from the last batch
            logits = model(batch_X)    #The model makes a guess (forward pass)
            loss = criterion(logits, batch_y)    #Calculate the loss (prediction - target) using Cross-Entropy loss
            loss.backward()    #Take derivative of loss
            optimizer.step()    #Update the weights based on the gradient
            total_loss += loss.item() * batch_X.size(0)    #batch_X.size(0) = batch_size = 64. Multiply average by num_items to get true loss
            total_items += batch_X.size(0)    #How many reviews have been processed so far

        _, val_metrics = evaluate(
            model,
            X_val,
            y_val,
            batch_size=batch_size,
            verbose=False,    #Don't print the validation metrics every training epoch
        )
        train_loss = total_loss / max(total_items, 1)    #This is an average. max prevents division by zero just in case
        print(
            f"\n Epoch {epoch + 1}/{epochs} - "
            f" Train loss: {train_loss:.4f} - val_f1: {val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_state = {
                key: value.detach().cpu().clone()    #Detach from the computational graph and copy to CPU
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)    #Return only the best model as the model
    return model


def evaluate(    #Run the model on the validation set and return a dictionary of metrics
    model,
    X_eval,
    y_eval,
    batch_size=128,    #Because we are not storing gradients as we are not going to update the weights, we can process more data at once
    verbose=True,
    model_name="Transformer Encoder",
):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_eval)),    #from_numpy: Convert the raw numpy arrays to PyTorch-compatible tensors
        batch_size=batch_size,
        shuffle=False,
    )
    preds = []

    with torch.no_grad():
        for (batch_X,) in loader:    #We don't collect batch_y because we did not pass y_eval to the DataLoader
            batch_X = batch_X.to(DEVICE)
            logits = model(batch_X)
            preds.append(logits.argmax(dim=1).cpu().numpy())    #Move the best result (argmax) from the GPU to the CPU and convert it to a numpy object

    y_pred = np.concatenate(preds)    #Join all the small, batch-sized predictions together into one big array
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
        "lr": 3e-3,
        "weight_decay": 1e-3,
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

    def objective(trial):    #This will be the mini-experiment Optuna will run
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
            "lr": trial.suggest_float("lr", 1e-3, 7e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        }

        vocab = build_vocab(train_texts, max_vocab=params["max_vocab"])
        X_train = texts_to_indices(train_texts, vocab, max_len=params["max_len"])
        X_val = texts_to_indices(val_texts, vocab, max_len=params["max_len"])

        model = build_model(len(vocab), params)    #Rebuild the Transformer with the new experimental parameters
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
        _, metrics = evaluate(    #We don't need the prediction data, so we ignore it
            model,
            X_val,
            y_val,
            batch_size=params["batch_size"],
            verbose=False,
        )

        return metrics["macro_f1"]    #This is the metric we are trying to maximize

    results = tune_model(    #This will run the Optuna hyperparameter tuning
        objective,
        n_trials=10,
        log_path=TUNING_LOG,
        model_name="Transformer Encoder",
    )
    save_best_config(results["best_config"], BEST_PARAMS_FILE)


def main(final=False):
    run_start = time.time()
    model_name = "Transformer Encoder"
    best, _ = load_best_config(BEST_PARAMS_FILE)
    params = merge_params(best)

    if best:
        print(f"Using tuned params: {best}")

    if final:
        print()
        with timed_step("Loading full dataset"):
            train_texts, y_train, _, _, eval_texts, y_eval = load_yelp_data(
                train_size=None,    #We are not training anymore. We want the unseen test set
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
        print()
        with timed_step("Loading dataset"):    #We are training and then evaluating with the validation set
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
    print(f"\nDone in {total:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--final", action="store_true", help="Full training set, evaluate on test")
    args = parser.parse_args()

    if args.tune:
        run_tuning()
    else:
        main(final=args.final)
