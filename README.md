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

Common flags:

- `--final` train on full 650k (default 150k subsample), evaluate on test set
- `--tune` run Optuna hyperparameter tuning
- `--discard` skip saving results (applies to both `results_log.md` and `tuning_log.md`)
- `--default` use default params instead of the tuned best config

Model-specific flags:

- **CNN:** `--(embedding option)` use pre-trained embeddings options such as GloVe 6B 100d (`--glove-6b-100d`)

Example:

```bash
python CNN/CNN.py --final
```

University of Central Florida, Spring 2026
