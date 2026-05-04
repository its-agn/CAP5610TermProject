# Results Log

## Decoder-only LLM (final, best config)

- **Date:** 2026-05-03 02:56
- **Device:** NVIDIA GeForce RTX 5070 Ti
- **Accuracy:** 0.6711
- **Macro Precision:** 0.6694
- **Macro Recall:** 0.6711
- **Macro F1:** 0.6701
- **Time:** 17006.2s (283.4m)

<details>
<summary>Config</summary>

**Params**
- `max_length`: 256
- `epochs`: 2
- `batch_size`: 2
- `eval_batch_size`: 8
- `learning_rate`: 1.278828922008836e-05
- `weight_decay`: 0.005334916687929385
- `warmup_ratio`: 0.017202394489085567
- `grad_accum_steps`: 1
- `patience`: 2

**Metadata**
- `model_name`: distilgpt2
- `seed`: 0
- `best_epoch`: 2
- `best_val_macro_f1`: 0.6708
- `resume_from`: pretrained

</details>
