# Term Project - CAP5610, Machine Learning

## Dataset

### [YelpReviewFull](https://huggingface.co/datasets/Yelp/yelp_review_full)

- _700k_ English Text Corpus of 2015 _Yelp reviews_ and associated ratings (1-5 stars)
- Dataset split into 650k training and 50k testing
- Each label perfectly split at 20% each on both train and test
- The dataset is publicly available through Hugging Face

## Objective

- We want to develop models capable of accurately predicting the number of stars associated to any given review, labeled 1 to 5.
- Depending on the tone and different keywords associated with specific ratings, our ideal models will match a text review to its appropriate star rating.

## Models Implemented

### Traditional:

- Linear SVM - Alexander Nardi
- Kernel SVM - Anthony Mahon
- Naïve Bayes Classifier - Benedetto Falin
- Decision Tree / Random Forest - Keating Sane
- Logistic Regression - Mohammed Mamdouh

### Deep Learning:

- Long Short-Term Memory - Alexander Nardi
- Transformer - Anthony Mahon
- Multi-Layer Perceptron - Benedetto Falin
- Convolutional Neural Network - Keating Sane
- Large Language Model - Mohammed Mamdouh

## Evaluation Metrics

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-Score
- Confusion Matrix

## How to Run

All models share common flags: `--final` (train on full 650k, evaluate on test set), `--tune` (Optuna hyperparameter tuning), and `--no-save` (skip writing to `results_log.md`). By default, models train on a 150k subsample with a validation split.

```bash
python Random_Forest/Random_Forest.py [--final] [--tune] [--no-save] [--single-tree]
python CNN/CNN.py                     [--final] [--tune] [--no-save] [--glove-6b | --no-glove]
```

Model-specific flags:
- **Random Forest:** `--single-tree` uses a single Decision Tree instead of the ensemble
- **CNN:** `--glove-6b` uses smaller GloVe 6B 100d embeddings; `--no-glove` uses random embeddings

University of Central Florida, Spring 2026
