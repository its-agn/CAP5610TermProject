# Random Forest Tuning Notes

Each search was run with Optuna for `20` trials with a `10%` validation holdout.

## Broad Search

The initial Random Forest search used a broad search grid over both TF-IDF and forest hyperparameters on `200000` training examples:

- `tfidf_features`: `[5000, 10000, 20000, 40000]`
- `ngram_max`: `1` to `2`
- `min_df`: `[3, 5, 10]`
- `max_df`: `[0.85, 0.9, 0.95]`
- `n_estimators`: `[100, 200, 300, 500]`
- `max_depth`: `[50, 100, 150, 300]`
- `min_samples_leaf`: `1` to `10`
- `max_features`: `["sqrt", "log2"]`

The broad search reached a validation Macro F1 of `0.5381`. The strongest Random Forest trials clustered around:

- `tfidf_features=20000`
- `ngram_max=2`
- `min_df=5` or `10`
- `max_df=0.85`
- `n_estimators=500`
- `max_depth=150` or `300`
- `min_samples_leaf=3` or `4`
- `max_features="sqrt"`

This suggests that the Random Forest benefits from bigram features, a moderate TF-IDF vocabulary size, and the stronger `sqrt` feature-subsampling rather than `log2`. The broad search also showed that unigram-only configurations and more weakly regularized forests were consistently worse.

## Focused Follow Up Search

After the broad search, the tuning space was narrowed to concentrate trials around the strongest region:

- `tfidf_features`: `[20000, 40000]`
- `ngram_max`: `[2]`
- `min_df`: `[5, 10]`
- `max_df`: `[0.85, 0.9]`
- `n_estimators`: `[500]`
- `max_depth`: `[150, 300]`
- `min_samples_leaf`: `[3, 4, 5]`
- `max_features`: `["sqrt"]`

This focused follow-up search was intended to test whether the broad-search winner was stable or whether a nearby configuration could improve on the `0.5381` result. The best Macro F1 score reached in this focused search was a `0.5386`, suggesting that this model has largely converged within this tuning space.
