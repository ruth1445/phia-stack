from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


def build_item_text(df: pd.DataFrame) -> pd.Series:
    """
    Combine columns into a single text field per item.
    You can tweak this later (add style_tags, material, etc.).
    """
    return df.apply(
        lambda r: f\"{r.get('title', '')} {r.get('description', '')} Brand: {r.get('brand', '')}\",
        axis=1,
    )


def encode_items(
    df: pd.DataFrame,
    model_name: str = \"all-MiniLM-L6-v2\",
) -> Tuple[SentenceTransformer, np.ndarray, pd.Series]:
    """
    Encode each item into an embedding using a sentence-transformers model.
    Returns:
      - model (so you can reuse it for queries)
      - item_embeddings (numpy array, shape [n_items, d])
      - corpus_text (Series of the text that was encoded)
    """
    model = SentenceTransformer(model_name)
    corpus_text = build_item_text(df)
    embeddings = model.encode(
        corpus_text.tolist(),
        normalize_embeddings=True,
    )
    return model, embeddings, corpus_text


def compute_user_taste_vector(
    item_embeddings: np.ndarray,
    df: pd.DataFrame,
    liked_ids: List[int],
) -> np.ndarray:
    """
    Build a user taste vector as the mean of embeddings for liked item_ids.
    """
    mask = df[\"item_id\"].isin(liked_ids)
    if not mask.any():
        raise ValueError(\"No liked_ids found in DataFrame.\")

    vec = item_embeddings[mask].mean(axis=0)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def compute_taste_scores(
    user_vec: np.ndarray,
    item_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between user taste vector and each item embedding.
    """
    sims = util.cos_sim(user_vec, item_embeddings).cpu().numpy().flatten()
    return sims


def add_taste_scores(
    df: pd.DataFrame,
    item_embeddings: np.ndarray,
    liked_ids: List[int],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Convenience helper:
      - builds user taste vector from liked_ids
      - computes taste_score for every item
      - returns (df_with_scores, user_vec)
    """
    user_vec = compute_user_taste_vector(item_embeddings, df, liked_ids)
    sims = compute_taste_scores(user_vec, item_embeddings)

    df_out = df.copy()
    df_out[\"taste_score\"] = sims
    return df_out, user_vec
