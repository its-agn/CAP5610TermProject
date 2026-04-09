# Results Log

## Decision Tree (validation, best config)

- **Date:** 2026-04-09 00:53
- **Device:** AMD Ryzen 7 9800X3D 8-Core Processor
- **Accuracy:** 0.3881
- **Macro Precision:** 0.3880
- **Macro Recall:** 0.3881
- **Macro F1:** 0.3879
- **Time:** 281.3s (4.7m)

<details>
<summary>Config</summary>

**Params**
- `tfidf_features`: 40000
- `ngram_max`: 2
- `min_df`: 10
- `max_df`: 0.9
- `max_depth`: 300
- `min_samples_leaf`: 10

**Metadata**
- `seed`: 0
- `tuning_train_size`: 200000
- `tuning_val_split`: 0.1
- `tuning_trials`: 20

</details>

## Decision Tree (final, best config)

- **Date:** 2026-04-09 01:54
- **Device:** AMD Ryzen 7 9800X3D 8-Core Processor
- **Accuracy:** 0.4082
- **Macro Precision:** 0.4074
- **Macro Recall:** 0.4082
- **Macro F1:** 0.4077
- **Time:** 3850.8s (64.2m)

<details>
<summary>Config</summary>

**Params**
- `tfidf_features`: 40000
- `ngram_max`: 2
- `min_df`: 10
- `max_df`: 0.9
- `max_depth`: 300
- `min_samples_leaf`: 10

**Metadata**
- `seed`: 0
- `tuning_train_size`: 200000
- `tuning_val_split`: 0.1
- `tuning_trials`: 20

</details>

## Decision Tree (validation, default config)

- **Date:** 2026-04-09 01:59
- **Device:** AMD Ryzen 7 9800X3D 8-Core Processor
- **Accuracy:** 0.3761
- **Macro Precision:** 0.3770
- **Macro Recall:** 0.3761
- **Macro F1:** 0.3762
- **Time:** 325.1s (5.4m)

<details>
<summary>Config</summary>

**Params**
- `tfidf_features`: 20000
- `ngram_max`: 2
- `min_df`: 5
- `max_df`: 0.9
- `max_depth`: 100
- `min_samples_leaf`: 5

**Metadata**
- `seed`: 0

</details>

## Decision Tree (final, default config)

- **Date:** 2026-04-09 03:11
- **Device:** AMD Ryzen 7 9800X3D 8-Core Processor
- **Accuracy:** 0.3949
- **Macro Precision:** 0.3950
- **Macro Recall:** 0.3949
- **Macro F1:** 0.3948
- **Time:** 4422.9s (73.7m)

<details>
<summary>Config</summary>

**Params**
- `tfidf_features`: 20000
- `ngram_max`: 2
- `min_df`: 5
- `max_df`: 0.9
- `max_depth`: 100
- `min_samples_leaf`: 5

**Metadata**
- `seed`: 0

</details>
