# Notebooks

Colab-first notebooks for the ML phases. Nothing is trained locally — every
notebook opens in Colab, reads data from the GitHub raw URL or the Hugging
Face Hub, logs experiments to Weights & Biases, and pushes artifacts back to
the Hub.

## Planned notebooks

| File | Purpose |
|---|---|
| `00_smoke_test.ipynb` | Load CSV from GitHub raw URL, log row count to W&B. Proves the cloud loop works. |
| `01_generate_synth_dataset.ipynb` | Run `data_synth/generate_synthetic_profiles.py`, push Parquet + dataset card to HF Datasets. |
| `02_baseline_eval.ipynb` | Score `data/test_profiles_10k.csv` pairs with the rule-based baseline; log NDCG@10 / Recall@50 / MAP to W&B. |
| `03_lightgbm_classifier.ipynb` | Pointwise binary classifier on pair features. |
| `04_lightgbm_lambdarank.ipynb` | Learning-to-rank with LightGBM LambdaRank. |
| `05_sentence_transformer_embeddings.ipynb` | Encode profile text, add cosine similarity as a feature. |
| `06_two_tower.ipynb` | (stretch) PyTorch two-tower model with in-batch negatives. |
| `07_eval_leaderboard.ipynb` | Final comparison table: baseline vs every model on the frozen test split. |

## Running a notebook

1. Open in Colab via the "Open in Colab" badge (add one at the top of each notebook).
2. Colab secrets needed: `WANDB_API_KEY`, `HF_TOKEN`, `ANTHROPIC_API_KEY`.
3. Save edits back with File → Save a copy in GitHub.
