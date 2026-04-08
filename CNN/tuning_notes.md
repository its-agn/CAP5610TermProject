# CNN Tuning Notes

Each search was run with Optuna for `30` trials with a `10%` validation holdout.

## Broad Search

The initial CNN search used a broad search grid over CNN hyperparameters on `150000` training examples:

- `num_filters`: `[100, 200, 300]`
- `filter_sizes`: `["3,4,5", "2,3,4,5"]`
- `dropout`: `0.2` to `0.6`
- `lr`: `1e-4` to `1e-2` (log scale)
- `batch_size`: `[64, 128]`
- `epochs`: `5` to `15`

Additionally, the broad search was run separately with six different embedding options and their resulting Macro F1 scores:

- `Random embeddings` (256 dimensions) -> `0.6123`
- `GloVe 6B 100d`: 90% coverage (100 dimensions) -> `0.6175`
- `GloVe 6B 300d`: 90% coverage (300 dimensions) -> `0.6204`
- `fastText wiki-news 300d subword`: 93% coverage (300 dimensions) -> `0.6076`
- `GloVe 2024 WikiGigaword 300d`: 93% coverage (300 dimensions) -> `0.6258`
- `GloVe 42B 300d`: 96% coverage (300 dimensions) -> `0.6228`

`GloVe 2024 WikiGigaword 300d` demonstrated that it was the strongest embedding option among the completed broad-search runs with a Macro F1 score of `0.6258`, despite not having the highest vocabulary coverage. This suggests that those pre-trained embeddings may be better aligned with more recent language usage. The strongest CNN trials clustered around:

- `num_filters=300`
- `filter_sizes="2,3,4,5"`
- lower learning rates in the low `1e-4` range
- `batch_size=128`
- moderate dropout rather than the highest tested values

This suggested that adding the size 2 filters was helpful and that the model preferred a relatively low learning rate with more convolution filters.

## Focused Follow Up Search

After the broad search, the tuning space was narrowed to concentrate trials around the best performing region on the full `650000` training split:

- `num_filters`: `[200, 300]`
- `filter_sizes`: `["2,3,4,5"]`
- `dropout`: `0.18` to `0.5`
- `lr`: `1e-4` to `6e-4` (log scale)
- `batch_size`: `[64, 128]`
- `epochs`: `5` to `10`

This focused follow-up search was then run with `GloVe 2024 WikiGigaword 300d` and reached a validation Macro F1 of `0.6440`.
