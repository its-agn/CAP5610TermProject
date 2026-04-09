# Results Log

## Random Forest (validation, best config)

- **Date:** 2026-04-08 21:02
- **Device:** AMD Ryzen 7 9800X3D 8-Core Processor
- **Accuracy:** 0.5479
- **Macro Precision:** 0.5380
- **Macro Recall:** 0.5479
- **Macro F1:** 0.5372
- **Time:** 180.3s (3.0m)

<details>
<summary>Config</summary>

**Params**
- `tfidf_features`: 40000
- `ngram_max`: 2
- `min_df`: 10
- `max_df`: 0.85
- `n_estimators`: 500
- `max_depth`: 150
- `min_samples_leaf`: 4
- `max_features`: sqrt

**Metadata**
- `seed`: 0
- `tuning_train_size`: 200000
- `tuning_val_split`: 0.1
- `tuning_trials`: 20

</details>

## Random Forest (final, best config)

- **Date:** 2026-04-08 21:45
- **Device:** AMD Ryzen 7 9800X3D 8-Core Processor
- **Accuracy:** 0.5555
- **Macro Precision:** 0.5464
- **Macro Recall:** 0.5555
- **Macro F1:** 0.5468
- **Time:** 2632.9s (43.9m)

<details>
<summary>Config</summary>

**Params**
- `tfidf_features`: 40000
- `ngram_max`: 2
- `min_df`: 10
- `max_df`: 0.85
- `n_estimators`: 500
- `max_depth`: 150
- `min_samples_leaf`: 4
- `max_features`: sqrt

**Metadata**
- `seed`: 0
- `tuning_train_size`: 200000
- `tuning_val_split`: 0.1
- `tuning_trials`: 20

</details>

## Random Forest (validation, default config)

- **Date:** 2026-04-08 21:47
- **Device:** AMD Ryzen 7 9800X3D 8-Core Processor
- **Accuracy:** 0.5287
- **Macro Precision:** 0.5178
- **Macro Recall:** 0.5287
- **Macro F1:** 0.5181
- **Time:** 122.0s (2.0m)

<details>
<summary>Config</summary>

**Params**
- `tfidf_features`: 20000
- `ngram_max`: 2
- `n_estimators`: 100
- `max_depth`: 150
- `min_samples_leaf`: 1
- `max_features`: sqrt
- `min_df`: 5
- `max_df`: 0.9

**Metadata**
- `seed`: 0

</details>

## Random Forest (final, default config)

- **Date:** 2026-04-08 22:13
- **Device:** AMD Ryzen 7 9800X3D 8-Core Processor
- **Accuracy:** 0.5361
- **Macro Precision:** 0.5261
- **Macro Recall:** 0.5361
- **Macro F1:** 0.5277
- **Time:** 1597.0s (26.6m)

<details>
<summary>Config</summary>

**Params**
- `tfidf_features`: 20000
- `ngram_max`: 2
- `n_estimators`: 100
- `max_depth`: 150
- `min_samples_leaf`: 1
- `max_features`: sqrt
- `min_df`: 5
- `max_df`: 0.9

**Metadata**
- `seed`: 0

</details>
