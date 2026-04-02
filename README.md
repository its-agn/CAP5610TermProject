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
- TBD

## AI Use:
- We acknowledge the use of AI to help set up, debug, and configure our models, helping us learn along the way.

University of Central Florida, Spring 2026
