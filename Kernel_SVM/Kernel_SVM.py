'''
Kernel Support Vector Machine (SVM) on YelpReviewFull dataset (5-class star rating prediction).
Anthony Mahon - CAP5610 Spring 2026

Run modes:
    python Kernel_SVM.py    #trains on 10k samples
    python Kernel_SVM.py --size n    #trains on n samples (no more than 20,000 (~9 mins on my machine) recommended)
    python Kernel_SVM.py --final    #runs on full test set
'''
import argparse
import os, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import time

#Tell Python to look one folder up for the utils package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import timed_step, load_yelp_data, compute_metrics, print_metrics, plot_confusion_matrix, save_results, tune_model, save_best_params

import logging    #Get rid of the annoying HuggingFace token warning
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.CRITICAL)


#Automated tuning of the hyperparameters
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TUNING_LOG = os.path.join(SCRIPT_DIR, "tuning_log.md")
BEST_PARAMS_FILE = os.path.join(SCRIPT_DIR, "best_params.json")


def run_tuning():
    #We load data once outside the objective function to save time, as optuna runs this function repeatedly (num_trials)
    train_texts, y_train, val_texts, y_val, _, _ = load_yelp_data(train_size=10000, val_split=0.2)

    def objective(trial):
        #Suggest hyperparameters
        c_val = trial.suggest_float("C", 0.1, 10.0, log=True)
        max_feats = trial.suggest_categorical("max_features", [5000, 10000, 15000])

        #Vectorize
        vectorizer = TfidfVectorizer(max_features=max_feats, stop_words='english')
        X_train = vectorizer.fit_transform(train_texts)
        X_val = vectorizer.transform(val_texts)

        #Train
        model = SVC(kernel='rbf', C=c_val)
        model.fit(X_train, y_train)

        #Evaluate
        y_pred = model.predict(X_val)
        metrics = compute_metrics(y_val, y_pred)
        return metrics["macro_f1"]

    results = (tune_model
        (
            objective,
            n_trials=15,
            log_path=TUNING_LOG,
            best_params_path=BEST_PARAMS_FILE,
            model_name="Kernel SVM"
        ))
    return results
#End of automated tuning


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run automated Optuna hyperparameter tuning")
    parser.add_argument("--size", type=int, default=10000, help="Number of training samples (default: 10000)")
    parser.add_argument("--final", action="store_true", help="Evaluate on the full test set")
    args = parser.parse_args()

    if args.tune:
        print("Running automated hyperparameter tuning... (this may take a while)")
        run_tuning()
        print(f"Tuning complete! Best params saved to {BEST_PARAMS_FILE}")
        return

    #Load and sub-sample the data
    with timed_step(f"Loading Yelp dataset (sub-sampled) (size={args.size})"):
        #This automatically subsamples to 10k to avoid training time issues!
        train_texts, y_train, val_texts, y_val, test_texts, test_labels = load_yelp_data(train_size=args.size)

        #If --final is typed, swap validation for the real test set
        final_string = None
        if args.final:
            val_texts, y_val = test_texts, test_labels
            final_string = "(full test set)"


    #Vectorize the input. We are going to scale and normalize to prevent distance distortion
    with timed_step("Applying TF-IDF Vectorization"):
        #max_features limits the dictionary to the top 10,000 most important words
        #Give highest scores to words that are frequent in a specific review, but rare across the whole dataset (e.g., "undercooked" or "phenomenal")
        vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')

        #Learn the vocabulary and transform the text into numbers
        X_train_vectors = vectorizer.fit_transform(train_texts)

        #Only transform, not (yet) fit, the test data from text to numbers that the computer model can use
        X_val_vectors = vectorizer.transform(val_texts)

    #Initialize and Train the Kernel SVM :)
    C = 1
    with timed_step(f"Training Kernel SVM (RBF) (C={C})"):
        #kernel='rbf' is the "Radial Basis Function" (the Kernel trick)
        #C=1.0 is the default penalty for mistakes, which ended up being the best value found for this dataset
        svm_model = SVC(kernel='rbf', C=C)

        #This is where the learning happens
        svm_model.fit(X_train_vectors, y_train)

    #Predict and Evaluate
    with timed_step(f"Making predictions and evaluating {final_string if final_string is not None else ""}"):
        y_pred = svm_model.predict(X_val_vectors)

        metrics = compute_metrics(y_val, y_pred)
        print_metrics(metrics, model_name="Kernel SVM", y_true=y_val, y_pred=y_pred)

    #Save the results
    #Define filenames
    results_file = os.path.join(os.path.dirname(__file__), "results_log.md")
    cm_file = os.path.join(os.path.dirname(__file__), "confusion_matrix_svm.png")

    #Save the plot
    plot_confusion_matrix(y_val, y_pred, cm_file, "Kernel SVM")

    #Save to the markdown log
    total_time = time.time() - start_time
    save_results("Kernel SVM", metrics, total_time, results_file, final=args.final)


if __name__ == "__main__":
    main()
