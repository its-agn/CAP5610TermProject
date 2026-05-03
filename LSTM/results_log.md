# Results Log

## LSTM (validation, default config)

- **Date:** 2026-05-02 20:13
- **Device:** NVIDIA GeForce RTX 4080 SUPER
- **Accuracy:** 0.6012
- **Macro Precision:** 0.5993
- **Macro Recall:** 0.6012
- **Macro F1:** 0.5986
- **Time:** 63.8s (1.1m)

<details>
<summary>Config</summary>

**Params**
- `max_vocab`: 50000
- `max_len`: 256
- `embed_dim`: 128
- `hidden_dim`: 128
- `num_layers`: 1
- `bidirectional`: True
- `dropout`: 0.3
- `lr`: 0.001
- `batch_size`: 128
- `epochs`: 5
- `patience`: 3

**Metadata**
- `seed`: 0
- `best_epoch`: 4
- `vocab_size`: 50000

</details>

## LSTM (final, best config)

- **Date:** 2026-05-02 23:01
- **Device:** NVIDIA GeForce RTX 4080 SUPER
- **Accuracy:** 0.6673
- **Macro Precision:** 0.6675
- **Macro Recall:** 0.6673
- **Macro F1:** 0.6673
- **Time:** 5067.8s (84.5m)

<details>
<summary>Config</summary>

**Params**
- `max_vocab`: 30000
- `max_len`: 512
- `embed_dim`: 300
- `hidden_dim`: 256
- `num_layers`: 2
- `bidirectional`: True
- `dropout`: 0.2767911832239808
- `lr`: 0.0005444120936367603
- `batch_size`: 64
- `epochs`: 4
- `patience`: 3

**Metadata**
- `seed`: 0
- `tuning_train_size`: 50000
- `tuning_val_split`: 0.1
- `tuning_trials`: 15

</details>
