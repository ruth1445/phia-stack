from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


def _compute_query_sims(
    model: SentenceTransformer,
    query: str,
    item_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Encode the query and compute cosine similarity with each item embedding.
    """
    q_emb = model.encode([query], normalize_embeddings=True)
    sims = util.cos_sim(q_emb, item_embeddings).cpu().numpy().flatten()
    return sims


def _minmax_normalize(x: np.ndarray) -> np.ndarray:
    x_min = x.min()
    x_max = x.max()
    if x_max <= x_min:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def naive_rank(
    df: pd.DataFrame,
    item_embeddings: np.ndarray,
    model: SentenceTransformer,
    query: str,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Naive ranking: sort items by query similarity only.
    Returns a new DataFrame with a 'query_sim' column.
    """
    q_sims = _compute_query_sims(model, query, item_embeddings)
    df_out = df.copy()
    df_out["query_sim"] = q_sims
    return df_out.sort_values("query_sim", ascending=False).head(top_k)


def bias_aware_rank(
    df: pd.DataFrame,
    item_embeddings: np.ndarray,
    model: SentenceTransformer,
    query: str,
    alpha: float = 0.4,  # weight on query similarity
    beta: float = 0.3,   # weight on taste_score
    gamma: float = 0.3,  # weight on smart_buy_index
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Bias-aware ranking:
      final_score = alpha * query_sim_norm
                  + beta  * taste_score_norm
                  + gamma * smart_buy_norm

    Assumes df already has:
      - 'taste_score'
      - 'smart_buy_index' (0–100)
    """
    # 1) query similarity
    q_sims = _compute_query_sims(model, query, item_embeddings)
    q_sims_norm = _minmax_normalize(q_sims)

    # 2) taste_score normalization
    taste = df["taste_score"].to_numpy()
    taste_norm = _minmax_normalize(taste)

    # 3) smart_buy normalization (already 0–100)
    smart = df["smart_buy_index"].to_numpy() / 100.0

    final_score = alpha * q_sims_norm + beta * taste_norm + gamma * smart

    df_out = df.copy()
    df_out["query_sim"] = q_sims_norm
    df_out["final_score"] = final_score

    return df_out.sort_values("final_score", ascending=False).head(top_k)
