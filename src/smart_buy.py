from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# You can tweak these later or even load from a config
BRAND_STRENGTH: Dict[str, float] = {
    "Zara": 0.6,
    "Banana Republic": 0.65,
    "Everlane": 0.8,
    "Unknown": 0.4,
}

MATERIAL_SCORE: Dict[str, float] = {
    "leather": 0.9,
    "wool": 0.85,
    "cotton": 0.7,
    "cotton blend": 0.65,
    "polyester": 0.4,
}

CONDITION_SCORE: Dict[str, float] = {
    "like new": 0.95,
    "very good": 0.85,
    "good": 0.7,
    "fair": 0.5,
}


def _compute_discount_ratio(price: float, original_price: float) -> float:
    if original_price is None or original_price <= 0:
        return 0.0
    disc = (original_price - price) / original_price
    return float(max(0.0, min(1.0, disc)))


def _compute_component_scores(row: pd.Series) -> Dict[str, float]:
    brand = str(row.get("brand", "") or "").strip()
    material = str(row.get("material", "") or "").lower().strip()
    condition_norm = row.get("condition_norm")

    discount_ratio = _compute_discount_ratio(
        price=float(row.get("price", 0.0)),
        original_price=float(row.get("original_price", 0.0)),
    )

    brand_score = BRAND_STRENGTH.get(brand, 0.5)
    material_score = MATERIAL_SCORE.get(material, 0.5)
    condition_score = CONDITION_SCORE.get(condition_norm, 0.6)

    return {
        "discount_ratio": discount_ratio,
        "brand_score": brand_score,
        "material_score": material_score,
        "condition_score": condition_score,
    }


def add_smart_buy_index(
    df: pd.DataFrame,
    w_disc: float = 0.4,
    w_brand: float = 0.2,
    w_mat: float = 0.2,
    w_cond: float = 0.2,
) -> pd.DataFrame:
    """
    Add Smart Buy components and a 0–100 Smart Buy Index to the DataFrame.

    New columns:
      - discount_ratio
      - brand_score
      - material_score
      - condition_score
      - smart_buy_index_raw
      - smart_buy_index  (scaled 0–100)
    """
    comp_records = df.apply(_compute_component_scores, axis=1, result_type="expand")
    df_out = pd.concat([df.reset_index(drop=True), comp_records], axis=1)

    df_out["smart_buy_index_raw"] = (
        w_disc * df_out["discount_ratio"] +
        w_brand * df_out["brand_score"] +
        w_mat * df_out["material_score"] +
        w_cond * df_out["condition_score"]
    )

    scaler = MinMaxScaler(feature_range=(0, 100))
    df_out["smart_buy_index"] = scaler.fit_transform(
        df_out[["smart_buy_index_raw"]]
    )

    return df_out
