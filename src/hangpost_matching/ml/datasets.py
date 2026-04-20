"""Dataset loaders and train/val/test splits for ML training.

Planned contents:
- load_profiles_parquet(path_or_hf_uri) -> pd.DataFrame
- load_pairs_parquet(path_or_hf_uri) -> pd.DataFrame
- make_splits(df, seed=42, test=0.1, val=0.1) -> (train, val, test)
- build_lightgbm_rank_dataset(pairs, features) -> lgb.Dataset with group sizes
"""
