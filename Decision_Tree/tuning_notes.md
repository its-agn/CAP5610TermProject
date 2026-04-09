# Decision Tree Tuning Notes

The Decision Tree uses a single broad Optuna search of `20` trials with a `10%` validation holdout. No focused follow-up search is planned for this model because the Decision Tree serves primarily as a baseline for the Random Forest and does not justify the additional tuning cost.

## Broad Search

The Decision Tree search uses a broad search grid over both TF-IDF and tree hyperparameters on `200000` training examples:

- `tfidf_features`: `[5000, 10000, 20000, 40000]`
- `ngram_max`: `1` to `2`
- `min_df`: `[3, 5, 10]`
- `max_df`: `[0.85, 0.9, 0.95]`
- `max_depth`: `[50, 100, 150, 300, None]`
- `min_samples_leaf`: `1` to `10`

The broad search reached a validation Macro F1 of `0.3914`. The strongest region favored bigrams, higher `min_df`, `max_df` of `0.9`, and larger `min_samples_leaf`.
