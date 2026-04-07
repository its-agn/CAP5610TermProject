"""Shared text feature extraction helpers."""

from .timer import timed_step

def fit_tfidf_features(train_texts, eval_texts, max_features=20000,
                       ngram_range=(1, 2), indent=""):
    """Fit TF-IDF on train_texts and transform eval_texts with the same vocabulary."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    with timed_step(f"{indent}Fitting TF-IDF on training data"):
        X_train = vectorizer.fit_transform(train_texts)
    with timed_step(f"{indent}Transforming eval data"):
        X_eval = vectorizer.transform(eval_texts)
    return X_train, X_eval
