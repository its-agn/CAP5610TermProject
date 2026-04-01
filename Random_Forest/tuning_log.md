# Random Forest - Hyperparameter Tuning Log

## Tuning decisions

Round 1 (54 combos) tested TF-IDF features [10k, 20k, 40k], ngrams [(1,1), (1,2)], estimators [50, 100, 200], max depth [50, 150, None]. Findings:
- 40k features never outperformed 20k (more noise, no gain)
- Bigrams (1,2) consistently beat unigrams (1,1) by 2-3%
- More trees always improved results (200 > 100 > 50)
- max_depth=50 was always worst; 150 vs None was negligible
- Best result: 20k features, (1,2), 200 trees, depth 150 -> 0.5300 Macro F1

Round 2 drops losers (40k features, unigrams, low tree counts, depth 50), adds trigrams (1,3), 300 trees, and min_samples_leaf regularization [1, 3, 5].

## Results

| # | Date | TF-IDF Features | Ngram | Estimators | Max Depth | Min Leaf | Train Size | Accuracy | Macro P | Macro R | Macro F1 | Time (s) |
|---|------|-----------------|-------|------------|-----------|----------|------------|----------|---------|---------|----------|----------|
