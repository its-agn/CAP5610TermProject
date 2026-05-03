"""
Decoder-only LLM sentiment classifier for Yelp Review Full.
Mohammed Mamdouh - CAP5610 Spring 2026

This implementation fine-tunes a pretrained decoder-only language model
for 5-class review rating prediction while reusing the shared repo utils
for data loading, tuning, evaluation, and result logging.

Example usage:
    python Large_Language_Model/Large_Language_Model.py
    python Large_Language_Model/Large_Language_Model.py --tune
    python Large_Language_Model/Large_Language_Model.py --final
    python Large_Language_Model/Large_Language_Model.py --resume
    python Large_Language_Model/Large_Language_Model.py --smoke --discard
"""

import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    DEFAULT_SEED,
    LABEL_NAMES,
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

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_CONFIG_FILE = os.path.join(SCRIPT_DIR, "best_config.json")
RESULTS_LOG = os.path.join(SCRIPT_DIR, "results_log.md")
TUNING_LOG = os.path.join(SCRIPT_DIR, "tuning_log.md")
CM_PATH = os.path.join(SCRIPT_DIR, "confusion_matrix_llm.png")
SAVED_MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_model")
SMOKE_MODEL_NAME = "sshleifer/tiny-gpt2"
DEFAULT_MODEL_NAME = "distilgpt2"
NUM_LABELS = 5
VALIDATION_TRAIN_SIZE = 20000
TUNING_TRAIN_SIZE = 6000
FINAL_VAL_SPLIT = 0.05
MAX_GRAD_NORM = 1.0
TUNING_TRIALS = 8

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


class EncodedTextDataset(Dataset):
    """Simple torch dataset backed by pre-tokenized inputs."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = int(self.labels[idx])
        return item


@dataclass
class TrainArtifacts:
    """Outputs from a train run that are useful for logging and saving."""

    model: AutoModelForSequenceClassification
    train_seconds: float
    best_val_f1: float
    best_epoch: int


def format_duration(seconds):
    """Format elapsed time as H:MM:SS or M:SS for progress output."""
    seconds = max(int(seconds), 0)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def parse_args():
    """Parse CLI options for validation, tuning, final training, and resume."""
    parser = common_parser()
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Decoder-only pretrained model to fine-tune (default: distilgpt2)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from Large_Language_Model/saved_model",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Optional path to a previously saved fine-tuned checkpoint",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny decoder-only model and a tiny subset for a quick sanity check",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Override the default non-final training subset size",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=TUNING_TRIALS,
        help=f"Number of Optuna trials to run during --tune (default: {TUNING_TRIALS})",
    )
    args = parser.parse_args()
    if args.tune and args.default:
        parser.error("--tune cannot be combined with --default")
    if args.tune and args.resume:
        parser.error("--tune cannot be combined with --resume")
    if args.tune and args.resume_from:
        parser.error("--tune cannot be combined with --resume-from")
    if args.final and args.smoke:
        parser.error("--final cannot be combined with --smoke")
    return args


def get_default_params():
    """Return a conservative default config for decoder-only fine-tuning."""
    batch_size = 8 if DEVICE.type == "cuda" else 2
    eval_batch_size = 16 if DEVICE.type == "cuda" else 4
    grad_accum_steps = 2 if DEVICE.type == "cuda" else 1
    return {
        "max_length": 256,
        "epochs": 2,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "grad_accum_steps": grad_accum_steps,
        "patience": 2,
    }


def resolve_run_settings(args):
    """Apply smoke-mode overrides and normalize training source settings."""
    params = get_default_params()
    model_name = args.model_name
    train_size = args.train_size if args.train_size is not None else VALIDATION_TRAIN_SIZE

    if args.smoke:
        model_name = SMOKE_MODEL_NAME
        train_size = 256
        params.update({
            "max_length": 128,
            "epochs": 1,
            "batch_size": 4,
            "eval_batch_size": 8,
            "learning_rate": 5e-4,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "grad_accum_steps": 1,
            "patience": 1,
        })

    return model_name, train_size, params


def merge_params(best_params):
    """Overlay persisted best params onto defaults."""
    params = get_default_params()
    if best_params:
        params.update(best_params)
    return params


def resolve_resume_source(args):
    """Return the checkpoint directory to resume from, if requested."""
    if args.resume_from:
        return args.resume_from
    if args.resume:
        return SAVED_MODEL_DIR
    return None


def load_tokenizer_and_model(model_name, num_labels=NUM_LABELS, resume_from=None):
    """Load tokenizer/model either from pretrained weights or a local checkpoint."""
    source = resume_from if resume_from else model_name
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    config = AutoConfig.from_pretrained(source)
    config.num_labels = num_labels
    config.id2label = {idx: label for idx, label in enumerate(LABEL_NAMES)}
    config.label2id = {label: idx for idx, label in enumerate(LABEL_NAMES)}
    model = AutoModelForSequenceClassification.from_pretrained(source, config=config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.num_labels = num_labels
    model.config.problem_type = "single_label_classification"
    return tokenizer, model


def tokenize_texts(texts, tokenizer, max_length, desc):
    """Tokenize texts in batches to avoid large temporary spikes."""
    input_ids = []
    attention_masks = []

    with timed_step(desc):
        for start in range(0, len(texts), 512):
            batch = texts[start:start + 512]
            encoded = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            input_ids.extend(encoded["input_ids"])
            attention_masks.extend(encoded["attention_mask"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
    }


def build_loader(texts, labels, tokenizer, batch_size, max_length, shuffle):
    """Build a DataLoader with dynamic padding."""
    encodings = tokenize_texts(
        texts,
        tokenizer,
        max_length=max_length,
        desc=f"Tokenizing {len(texts):,} texts",
    )
    dataset = EncodedTextDataset(encodings, labels)
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if DEVICE.type == "cuda" else None,
        return_tensors="pt",
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )


def evaluate_model(model, loader, model_name="Decoder-only LLM", verbose=True):
    """Run inference and compute shared repo metrics."""
    model.eval()
    preds = []
    labels = []

    with timed_step("Running evaluation"):
        with torch.no_grad():
            for batch in loader:
                labels.append(batch["labels"].numpy())
                batch = {key: value.to(DEVICE) for key, value in batch.items()}
                logits = model(**batch).logits
                preds.append(logits.argmax(dim=-1).cpu().numpy())

    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    metrics = compute_metrics(y_true, y_pred)
    if verbose:
        print_metrics(metrics, model_name, y_true=y_true, y_pred=y_pred)
    return y_true, y_pred, metrics


def save_model_checkpoint(model, tokenizer, save_dir, params, metadata):
    """Persist the best checkpoint so training can be resumed later."""
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"params": params, "metadata": metadata}, f, indent=2)
    print(f"Saved checkpoint to {save_dir}")


def train_model(
    model,
    tokenizer,
    train_loader,
    val_loader,
    params,
    checkpoint_dir=None,
    checkpoint_metadata=None,
):
    """Fine-tune a decoder-only classifier with early stopping on validation macro-F1."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    total_optimization_steps = math.ceil(
        len(train_loader) * params["epochs"] / max(params["grad_accum_steps"], 1)
    )
    warmup_steps = int(total_optimization_steps * params["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimization_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_score = -1.0
    best_epoch = 0
    best_state = None
    patience_left = params["patience"]
    train_start = time.monotonic()
    epoch_durations = []
    epoch_bar = tqdm(
        range(1, params["epochs"] + 1),
        desc="Epochs",
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        sample_count = 0
        epoch_start = time.monotonic()
        batch_bar = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{params['epochs']}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

        for step, batch in batch_bar:
            labels = batch["labels"]
            sample_count += labels.size(0)
            batch = {key: value.to(DEVICE) for key, value in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
                outputs = model(**batch)
                loss = outputs.loss / params["grad_accum_steps"]

            scaler.scale(loss).backward()
            running_loss += loss.item() * labels.size(0) * params["grad_accum_steps"]

            if step % params["grad_accum_steps"] == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            batch_bar.set_postfix(
                loss=f"{running_loss / max(sample_count, 1):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        batch_bar.close()

        _, _, val_metrics = evaluate_model(model, val_loader, verbose=False)
        train_loss = running_loss / max(sample_count, 1)
        elapsed = time.monotonic() - epoch_start
        epoch_durations.append(elapsed)
        avg_epoch_seconds = sum(epoch_durations) / len(epoch_durations)
        remaining_epochs = params["epochs"] - epoch
        eta_seconds = avg_epoch_seconds * remaining_epochs
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_f1=f"{val_metrics['macro_f1']:.4f}",
            best_f1=f"{max(best_score, val_metrics['macro_f1']):.4f}",
            epoch_time=format_duration(elapsed),
            eta=format_duration(eta_seconds),
        )

        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_left = params["patience"]
            if checkpoint_dir and checkpoint_metadata:
                save_model_checkpoint(model, tokenizer, checkpoint_dir, params, checkpoint_metadata)
        else:
            patience_left -= 1
            if patience_left <= 0:
                tqdm.write(f"Early stopping at epoch {epoch} (no macro-F1 improvement)")
                break

    epoch_bar.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainArtifacts(
        model=model,
        train_seconds=time.monotonic() - train_start,
        best_val_f1=best_score,
        best_epoch=best_epoch,
    )


def run_tuning(args):
    """Tune only training hyperparameters while keeping the decoder model fixed."""
    set_random_seed(DEFAULT_SEED)
    model_name, _, base_params = resolve_run_settings(args)
    print_run_header(
        "Decoder-only LLM",
        mode="tuning",
        device=get_device_name(),
        seed=DEFAULT_SEED,
        extra_info={
            "Model": model_name,
            "Train subset": TUNING_TRAIN_SIZE,
            "Trials": args.trials,
        },
    )

    with timed_step("Loading tuning dataset"):
        train_texts, y_train, val_texts, y_val, _, _ = load_yelp_data(
            train_size=TUNING_TRAIN_SIZE,
            val_split=0.1,
            skip_test=True,
            seed=DEFAULT_SEED,
        )
    assert val_texts is not None and y_val is not None

    tokenizer, _ = load_tokenizer_and_model(model_name)

    def objective(trial):
        set_random_seed(DEFAULT_SEED)
        params = dict(base_params)
        params.update({
            "max_length": trial.suggest_categorical("max_length", [128, 192, 256]),
            "epochs": trial.suggest_categorical("epochs", [1, 2]),
            "batch_size": trial.suggest_categorical("batch_size", [2, 4]),
            "eval_batch_size": trial.suggest_categorical("eval_batch_size", [4, 8]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 8e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.12),
            "grad_accum_steps": trial.suggest_categorical("grad_accum_steps", [1, 2, 4]),
            "patience": 1,
        })

        train_loader = build_loader(
            train_texts,
            y_train,
            tokenizer,
            batch_size=params["batch_size"],
            max_length=params["max_length"],
            shuffle=True,
        )
        val_loader = build_loader(
            val_texts,
            y_val,
            tokenizer,
            batch_size=params["eval_batch_size"],
            max_length=params["max_length"],
            shuffle=False,
        )
        _, model = load_tokenizer_and_model(model_name)
        artifacts = train_model(model, tokenizer, train_loader, val_loader, params)
        _, _, metrics = evaluate_model(artifacts.model, val_loader, verbose=False)
        trial.set_user_attr("best_epoch", artifacts.best_epoch)
        return metrics["macro_f1"]

    results = tune_model(
        objective,
        n_trials=args.trials,
        log_path=None if args.discard else TUNING_LOG,
        model_name="Decoder-only LLM",
        extra_info={"Model": model_name, "Train subset": TUNING_TRAIN_SIZE},
        seed=DEFAULT_SEED,
    )
    if not args.discard:
        best_config = dict(results["best_config"])
        best_epoch = results["best_user_attrs"].get("best_epoch")
        if best_epoch:
            best_config["epochs"] = best_epoch
        save_best_config(
            best_config,
            BEST_CONFIG_FILE,
            metadata={"model_name": model_name, "seed": DEFAULT_SEED},
            macro_f1=results["best_score"],
        )


def prepare_datasets(args, params):
    """Load dataset splits for validation or final runs."""
    model_name, train_size, _ = resolve_run_settings(args)
    if args.final:
        with timed_step("Loading full dataset"):
            train_texts, y_train, _, _, eval_texts, y_eval = load_yelp_data(
                train_size=None,
                val_split=0,
                seed=DEFAULT_SEED,
            )
        train_texts, val_texts, y_train, y_val = train_test_split(
            train_texts,
            y_train,
            test_size=FINAL_VAL_SPLIT,
            stratify=y_train,
            random_state=DEFAULT_SEED,
        )
        eval_label = "test"
    else:
        with timed_step("Loading dataset"):
            train_texts, y_train, val_texts, y_val, _, _ = load_yelp_data(
                train_size=train_size,
                val_split=0.1,
                skip_test=True,
                seed=DEFAULT_SEED,
            )
        eval_texts = val_texts
        y_eval = y_val
        eval_label = "validation"

    assert val_texts is not None and y_val is not None
    assert eval_texts is not None and y_eval is not None
    print(
        f"Train: {len(train_texts):,} | "
        f"Val: {len(val_texts):,} | "
        f"Eval ({eval_label}): {len(eval_texts):,} | "
        f"Device: {DEVICE}"
    )
    return model_name, train_texts, y_train, val_texts, y_val, eval_texts, y_eval, eval_label


def main(args):
    """Run validation/final training, evaluation, optional checkpoint resume, and result logging."""
    set_random_seed(DEFAULT_SEED)
    run_start = time.monotonic()
    saved_params, saved_metadata = (None, {}) if args.default else load_best_config(BEST_CONFIG_FILE)
    model_name, _, fallback_params = resolve_run_settings(args)
    params = fallback_params if args.smoke else merge_params(saved_params)
    if args.smoke:
        params_source = "smoke config"
    elif saved_params and not args.default:
        params_source = "best tuned config"
    else:
        params_source = "default config"

    resume_source = resolve_resume_source(args)
    if saved_metadata.get("model_name") and not args.default and not args.smoke:
        model_name = saved_metadata["model_name"]
    if resume_source and not os.path.exists(resume_source):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_source}")

    print_run_header(
        "Decoder-only LLM",
        mode="final" if args.final else "validation",
        device=get_device_name(),
        seed=DEFAULT_SEED,
        extra_info={
            "Base model": model_name,
            "Config source": params_source,
            "Resume": resume_source if resume_source else "fresh pretrained weights",
        },
    )
    print_value_section("Parameters", params)

    (
        model_name,
        train_texts,
        y_train,
        val_texts,
        y_val,
        eval_texts,
        y_eval,
        eval_label,
    ) = prepare_datasets(args, params)

    tokenizer, model = load_tokenizer_and_model(
        model_name=model_name,
        resume_from=resume_source,
    )
    train_loader = build_loader(
        train_texts,
        y_train,
        tokenizer,
        batch_size=params["batch_size"],
        max_length=params["max_length"],
        shuffle=True,
    )
    val_loader = build_loader(
        val_texts,
        y_val,
        tokenizer,
        batch_size=params["eval_batch_size"],
        max_length=params["max_length"],
        shuffle=False,
    )
    eval_loader = build_loader(
        eval_texts,
        y_eval,
        tokenizer,
        batch_size=params["eval_batch_size"],
        max_length=params["max_length"],
        shuffle=False,
    )

    checkpoint_metadata = {
        "model_name": model_name,
        "seed": DEFAULT_SEED,
        "final": args.final,
        "eval_split": eval_label,
    }
    checkpoint_dir = None if args.discard else SAVED_MODEL_DIR

    artifacts = train_model(
        model,
        tokenizer,
        train_loader,
        val_loader,
        params,
        checkpoint_dir=checkpoint_dir,
        checkpoint_metadata=checkpoint_metadata,
    )

    y_true, y_pred, metrics = evaluate_model(
        artifacts.model,
        eval_loader,
        model_name="Decoder-only LLM",
    )

    total_time = time.monotonic() - run_start
    if not args.discard:
        saved = save_results(
            "Decoder-only LLM",
            metrics,
            total_time,
            RESULTS_LOG,
            final=args.final,
            device=get_device_name(),
            default_config=(params_source == "default config"),
            params=params,
            metadata={
                "model_name": model_name,
                "seed": DEFAULT_SEED,
                "best_epoch": artifacts.best_epoch,
                "best_val_macro_f1": round(artifacts.best_val_f1, 4),
                "resume_from": resume_source or "pretrained",
            },
        )
        if saved:
            plot_confusion_matrix(
                y_true,
                y_pred,
                CM_PATH,
                "Decoder-only LLM",
                title_suffix=f"Macro F1={metrics['macro_f1']:.4f}",
            )
    else:
        print("Skipping results save (--discard)")

    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f}m)")


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.tune:
        run_tuning(cli_args)
    else:
        main(cli_args)
