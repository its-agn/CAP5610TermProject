# Results Log

## TextCNN | GloVe 2024 WikiGigaword 300d (validation, best config)

- **Date:** 2026-04-08 17:10
- **Device:** NVIDIA GeForce RTX 5090
- **Accuracy:** 0.6250
- **Macro Precision:** 0.6201
- **Macro Recall:** 0.6250
- **Macro F1:** 0.6198
- **Time:** 57.2s (1.0m)

<details>
<summary>Config</summary>

**Params**
- `num_filters`: 300
- `filter_sizes`: 2,3,4,5
- `dropout`: 0.3454305242971086
- `lr`: 0.0003414059395182117
- `batch_size`: 128
- `epochs`: 3

**Metadata**
- `embedding_source`: 2024WG
- `embedding_dim`: 300
- `seed`: 0
- `tuning_train_size`: 650000
- `tuning_val_split`: 0.1
- `tuning_trials`: 30

</details>

## TextCNN | GloVe 2024 WikiGigaword 300d (final, best config)

- **Date:** 2026-04-08 17:13
- **Device:** NVIDIA GeForce RTX 5090
- **Accuracy:** 0.6422
- **Macro Precision:** 0.6412
- **Macro Recall:** 0.6422
- **Macro F1:** 0.6408
- **Time:** 226.4s (3.8m)

<details>
<summary>Config</summary>

**Params**
- `num_filters`: 300
- `filter_sizes`: 2,3,4,5
- `dropout`: 0.3454305242971086
- `lr`: 0.0003414059395182117
- `batch_size`: 128
- `epochs`: 3

**Metadata**
- `embedding_source`: 2024WG
- `embedding_dim`: 300
- `seed`: 0
- `tuning_train_size`: 650000
- `tuning_val_split`: 0.1
- `tuning_trials`: 30

</details>

## TextCNN | Random embeddings (validation, default config)

- **Date:** 2026-04-08 17:14
- **Device:** NVIDIA GeForce RTX 5090
- **Accuracy:** 0.5853
- **Macro Precision:** 0.5769
- **Macro Recall:** 0.5853
- **Macro F1:** 0.5785
- **Time:** 58.3s (1.0m)

<details>
<summary>Config</summary>

**Params**
- `num_filters`: 100
- `filter_sizes`: 3,4,5
- `dropout`: 0.5
- `lr`: 0.001
- `batch_size`: 64
- `epochs`: 10

**Metadata**
- `embedding_source`: random
- `embedding_dim`: 256
- `seed`: 0

</details>

## TextCNN | Random embeddings (final, default config)

- **Date:** 2026-04-08 17:21
- **Device:** NVIDIA GeForce RTX 5090
- **Accuracy:** 0.6090
- **Macro Precision:** 0.6085
- **Macro Recall:** 0.6090
- **Macro F1:** 0.6074
- **Time:** 411.7s (6.9m)

<details>
<summary>Config</summary>

**Params**
- `num_filters`: 100
- `filter_sizes`: 3,4,5
- `dropout`: 0.5
- `lr`: 0.001
- `batch_size`: 64
- `epochs`: 10

**Metadata**
- `embedding_source`: random
- `embedding_dim`: 256
- `seed`: 0

</details>

## TextCNN | GloVe 6B 300d (validation, default config)

- **Date:** 2026-04-08 17:22
- **Device:** NVIDIA GeForce RTX 5090
- **Accuracy:** 0.5985
- **Macro Precision:** 0.5950
- **Macro Recall:** 0.5985
- **Macro F1:** 0.5947
- **Time:** 59.8s (1.0m)

<details>
<summary>Config</summary>

**Params**
- `num_filters`: 100
- `filter_sizes`: 3,4,5
- `dropout`: 0.5
- `lr`: 0.001
- `batch_size`: 64
- `epochs`: 10

**Metadata**
- `embedding_source`: 6B
- `embedding_dim`: 300
- `seed`: 0

</details>

## TextCNN | GloVe 6B 300d (final, default config)

- **Date:** 2026-04-08 17:29
- **Device:** NVIDIA GeForce RTX 5090
- **Accuracy:** 0.6088
- **Macro Precision:** 0.6140
- **Macro Recall:** 0.6088
- **Macro F1:** 0.6109
- **Time:** 424.7s (7.1m)

<details>
<summary>Config</summary>

**Params**
- `num_filters`: 100
- `filter_sizes`: 3,4,5
- `dropout`: 0.5
- `lr`: 0.001
- `batch_size`: 64
- `epochs`: 10

**Metadata**
- `embedding_source`: 6B
- `embedding_dim`: 300
- `seed`: 0

</details>

## TextCNN | GloVe 6B 100d (validation, default config)

- **Date:** 2026-04-08 17:30
- **Device:** NVIDIA GeForce RTX 5090
- **Accuracy:** 0.6032
- **Macro Precision:** 0.5989
- **Macro Recall:** 0.6032
- **Macro F1:** 0.5982
- **Time:** 61.3s (1.0m)

<details>
<summary>Config</summary>

**Params**
- `num_filters`: 100
- `filter_sizes`: 3,4,5
- `dropout`: 0.5
- `lr`: 0.001
- `batch_size`: 64
- `epochs`: 10

**Metadata**
- `embedding_source`: 6B
- `embedding_dim`: 100
- `seed`: 0

</details>

## TextCNN | GloVe 6B 100d (final, default config)

- **Date:** 2026-04-08 17:36
- **Device:** NVIDIA GeForce RTX 5090
- **Accuracy:** 0.6199
- **Macro Precision:** 0.6178
- **Macro Recall:** 0.6199
- **Macro F1:** 0.6180
- **Time:** 373.4s (6.2m)

<details>
<summary>Config</summary>

**Params**
- `num_filters`: 100
- `filter_sizes`: 3,4,5
- `dropout`: 0.5
- `lr`: 0.001
- `batch_size`: 64
- `epochs`: 10

**Metadata**
- `embedding_source`: 6B
- `embedding_dim`: 100
- `seed`: 0

</details>

## TextCNN | GloVe 42B 300d (validation, default config)

- **Date:** 2026-04-08 17:37
- **Device:** NVIDIA GeForce RTX 5090
- **Accuracy:** 0.6069
- **Macro Precision:** 0.6051
- **Macro Recall:** 0.6069
- **Macro F1:** 0.6030
- **Time:** 58.2s (1.0m)

<details>
<summary>Config</summary>

**Params**
- `num_filters`: 100
- `filter_sizes`: 3,4,5
- `dropout`: 0.5
- `lr`: 0.001
- `batch_size`: 64
- `epochs`: 10

**Metadata**
- `embedding_source`: 42B
- `embedding_dim`: 300
- `seed`: 0

</details>

## TextCNN | GloVe 42B 300d (final, default config)

- **Date:** 2026-04-08 17:44
- **Device:** NVIDIA GeForce RTX 5090
- **Accuracy:** 0.6084
- **Macro Precision:** 0.6110
- **Macro Recall:** 0.6084
- **Macro F1:** 0.6095
- **Time:** 435.0s (7.3m)

<details>
<summary>Config</summary>

**Params**
- `num_filters`: 100
- `filter_sizes`: 3,4,5
- `dropout`: 0.5
- `lr`: 0.001
- `batch_size`: 64
- `epochs`: 10

**Metadata**
- `embedding_source`: 42B
- `embedding_dim`: 300
- `seed`: 0

</details>

## TextCNN | GloVe 2024 WikiGigaword 300d (validation, default config)

- **Date:** 2026-04-08 17:46
- **Device:** NVIDIA GeForce RTX 5090
- **Accuracy:** 0.6071
- **Macro Precision:** 0.6045
- **Macro Recall:** 0.6071
- **Macro F1:** 0.6039
- **Time:** 84.9s (1.4m)

<details>
<summary>Config</summary>

**Params**
- `num_filters`: 100
- `filter_sizes`: 3,4,5
- `dropout`: 0.5
- `lr`: 0.001
- `batch_size`: 64
- `epochs`: 10

**Metadata**
- `embedding_source`: 2024WG
- `embedding_dim`: 300
- `seed`: 0

</details>

## TextCNN | GloVe 2024 WikiGigaword 300d (final, default config)

- **Date:** 2026-04-08 17:53
- **Device:** NVIDIA GeForce RTX 5090
- **Accuracy:** 0.6128
- **Macro Precision:** 0.6127
- **Macro Recall:** 0.6128
- **Macro F1:** 0.6127
- **Time:** 437.1s (7.3m)

<details>
<summary>Config</summary>

**Params**
- `num_filters`: 100
- `filter_sizes`: 3,4,5
- `dropout`: 0.5
- `lr`: 0.001
- `batch_size`: 64
- `epochs`: 10

**Metadata**
- `embedding_source`: 2024WG
- `embedding_dim`: 300
- `seed`: 0

</details>

## TextCNN | fastText wiki-news subword 300d (validation, default config)

- **Date:** 2026-04-08 17:54
- **Device:** NVIDIA GeForce RTX 5090
- **Accuracy:** 0.5947
- **Macro Precision:** 0.6103
- **Macro Recall:** 0.5947
- **Macro F1:** 0.5975
- **Time:** 63.6s (1.1m)

<details>
<summary>Config</summary>

**Params**
- `num_filters`: 100
- `filter_sizes`: 3,4,5
- `dropout`: 0.5
- `lr`: 0.001
- `batch_size`: 64
- `epochs`: 10

**Metadata**
- `embedding_source`: FT-WIKI-SUBWORD
- `embedding_dim`: 300
- `seed`: 0

</details>

## TextCNN | fastText wiki-news subword 300d (final, default config)

- **Date:** 2026-04-08 18:01
- **Device:** NVIDIA GeForce RTX 5090
- **Accuracy:** 0.6035
- **Macro Precision:** 0.6052
- **Macro Recall:** 0.6035
- **Macro F1:** 0.6037
- **Time:** 432.8s (7.2m)

<details>
<summary>Config</summary>

**Params**
- `num_filters`: 100
- `filter_sizes`: 3,4,5
- `dropout`: 0.5
- `lr`: 0.001
- `batch_size`: 64
- `epochs`: 10

**Metadata**
- `embedding_source`: FT-WIKI-SUBWORD
- `embedding_dim`: 300
- `seed`: 0

</details>
