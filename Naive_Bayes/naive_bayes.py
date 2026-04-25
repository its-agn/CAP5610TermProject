import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

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
    LABEL_NAMES,
    common_parser,
    load_best_config,
    save_best_config,
)

SEED = 0
MODEL_NAME = "Naive Bayes"
BEST_CFG = "nb_best_config.json"
RESULTS_LOG = "results_log.md"
CM_PATH = "nb_confusion_matrix.png"
SWEEP_PATH = "nb_alpha_sweep.png"

set_random_seed(SEED)

args = common_parser().parse_args()

device = get_device_name(cpu_only=True)
print_run_header(MODEL_NAME, device=device, seed=SEED)


# load data: use full training set if --final, smaller subset otherwise
if args.final:
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_yelp_data(
        train_size=None, val_split=0.1
    )
else:
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_yelp_data(
        train_size=150000, val_split=0.1, skip_test=True
    )

# fit tfidf on train, transform val (and test if running final)
eval_texts = test_texts if args.final else val_texts
X_train, X_eval = fit_tfidf_features(
    train_texts, eval_texts,
    max_features=20000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
)
eval_labels = test_labels if args.final else val_labels

print(f"X_train : {X_train.shape}")
print(f"X_eval  : {X_eval.shape}\n")


# alpha sweep: runs if --tune flag is passed or no saved config exists
saved_params, _ = load_best_config(BEST_CFG)
run_sweep = args.tune or (saved_params is None) or args.default

if run_sweep and not args.default:
    alphas         = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
    val_accuracies = []

    print("sweeping alpha values on val set...")
    print("-" * 42)

    for alpha in alphas:
        model = ComplementNB(alpha=alpha)
        model.fit(X_train, train_labels)
        val_preds = model.predict(X_eval)

        from sklearn.metrics import accuracy_score
        val_acc = accuracy_score(eval_labels, val_preds)
        val_accuracies.append(val_acc)

        print(f"  alpha={alpha:<6} | val acc: {val_acc:.4f}")

    best_alpha = alphas[np.argmax(val_accuracies)]
    print(f"\nbest alpha: {best_alpha}  (val acc: {max(val_accuracies):.4f})")

    if not args.discard:
        save_best_config({"alpha": best_alpha}, BEST_CFG)

    # save the sweep plot
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, val_accuracies, marker="o", linewidth=2, color="steelblue")
    plt.axvline(x=best_alpha, color="red", linestyle="--", label=f"best alpha = {best_alpha}")
    plt.xlabel("Alpha (smoothing parameter)")
    plt.ylabel("Validation Accuracy")
    plt.title("Complement Naive Bayes -- Alpha Sweep")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SWEEP_PATH, dpi=150)
    plt.close()
    print(f"saved: {SWEEP_PATH}")

else:
    # use saved or default alpha
    if saved_params and not args.default:
        best_alpha = saved_params["alpha"]
        print(f"using saved best alpha: {best_alpha}")
    else:
        best_alpha = 30.0
        print(f"using default alpha: {best_alpha}")


# train final model with best alpha
print(f"\ntraining with alpha={best_alpha}...")
with timed_step("Training ComplementNB"):
    final_model = ComplementNB(alpha=best_alpha)
    final_model.fit(X_train, train_labels)


# evaluate
with timed_step("Evaluating"):
    preds = final_model.predict(X_eval)

metrics = compute_metrics(eval_labels, preds)
split   = "test" if args.final else "val"
print_metrics(metrics, MODEL_NAME, y_true=eval_labels, y_pred=preds)

# confusion matrix
plot_confusion_matrix(
    eval_labels, preds,
    save_path=CM_PATH,
    model_name=MODEL_NAME,
    title_suffix=f"alpha={best_alpha} | {split} set",
)

# save results to log
if not args.discard:
    save_results(
        model_name=MODEL_NAME,
        metrics=metrics,
        elapsed=0,
        log_path=RESULTS_LOG,
        final=args.final,
        device=device,
        default_config=args.default,
        params={"alpha": best_alpha},
        metadata={"split": split, "tfidf_features": 20000, "ngram_range": "(1,2)"},
    )