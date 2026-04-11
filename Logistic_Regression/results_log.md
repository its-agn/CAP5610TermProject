# Results Log

## Logistic Regression (validation)

- **Date:** 2026-04-05 23:19
- **Accuracy:** 0.5945
- **Macro Precision:** 0.5925
- **Macro Recall:** 0.5945
- **Macro F1:** 0.5934
- **Time:** 67.4s (1.1m)

## Logistic Regression (final)

- **Date:** 2026-04-05 23:38
- **Accuracy:** 0.6168
- **Macro Precision:** 0.6146
- **Macro Recall:** 0.6168
- **Macro F1:** 0.6156
- **Time:** 1034.7s (17.2m)

## Logistic Regression (validation, best config)

- **Date:** 2026-04-10 18:08
- **Device:** 11th Gen Intel(R) Core(TM) i3-1115G4 @ 3.00GHz
- **Accuracy:** 0.6119
- **Macro Precision:** 0.6079
- **Macro Recall:** 0.6119
- **Macro F1:** 0.6094
- **Time:** 366.1s (6.1m)

<details>
<summary>Config</summary>

**Params**
- `tfidf_features`: 40000
- `ngram_max`: 2
- `min_df`: 2
- `max_df`: 0.9
- `C`: 0.3791873671893262
- `penalty`: l2
- `max_iter`: 300
- `tol`: 0.0001

**Metadata**
- `seed`: 0
- `tuning_train_size`: 100000
- `tuning_val_split`: 0.1
- `tuning_trials`: 6
- `tuning_status`: best completed trial from a timed-out tuning run

</details>
