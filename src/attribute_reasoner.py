from __future__ import annotations

from typing import List, Optional, Dict, Any
import pandas as pd


# --- vocab lists (you can expand these later) ---

COLOR_WORDS: List[str] = [
    "black", "white", "cream", "red", "brown", "blue", "green",
    "beige", "tan", "navy", "grey", "gray", "pink", "yellow",
    "purple", "orange"
]

CATEGORY_WORDS: List[str] = [
    "boots", "boot", "coat", "jacket", "dress", "trousers", "pants",
    "jeans", "skirt", "top", "blouse", "shirt", "sweater", "hoodie"
]

STYLE_WORDS: Dict[str, List[str]] = {
    "minimalist": ["minimal", "minimalist", "simple", "clean"],
    "vintage": ["vintage", "retro"],
    "party": ["partywear", "party", "night out", "club"],
    "workwear": ["workwear", "office", "tailored"],
    "statement": ["statement", "bold"],
    "classic": ["classic", "timeless"],
    "streetwear": ["streetwear", "casual", "relaxed"],
}

CONDITION_MAP: Dict[str, List[str]] = {
    "like new": ["like new", "new with tags", "nwt", "excellent"],
    "very good": ["very good", "vgc"],
    "good": ["good", "gently worn", "lightly worn"],
    "fair": ["fair", "worn", "used"],
}


# --- helper extractors ---


def _extract_first_match(text: str, vocab: List[str]) -> Optional[str]:
    text_l = text.lower()
    for token in vocab:
        if token in text_l:
            return token
    return None


def extract_color(text: str) -> Optional[str]:
    return _extract_first_match(text, COLOR_WORDS)


def extract_category(text: str) -> Optional[str]:
    raw = _extract_first_match(text, CATEGORY_WORDS)
    if raw is None:
        return None
    # normalize singular/plural variants
    if raw.endswith("s"):
        return raw.rstrip("s")
    return raw


def extract_style_tags(text: str) -> Optional[List[str]]:
    text_l = text.lower()
    found: List[str] = []
    for label, words in STYLE_WORDS.items():
        for w in words:
            if w in text_l:
                found.append(label)
                break
    return found or None


def normalize_condition(text: str) -> Optional[str]:
    text_l = text.lower()
    for label, words in CONDITION_MAP.items():
        for w in words:
            if w in text_l:
                return label
    return None


# --- main public API ---


def infer_attributes_from_row(row: pd.Series) -> Dict[str, Any]:
    """
    Given a row with at least:
      - title
      - description
      - condition_note (optional)
      - category_hint (optional)
    return a dict of structured attributes.
    """
    title = str(row.get("title", ""))
    desc = str(row.get("description", ""))
    cond = str(row.get("condition_note", ""))
    cat_hint = row.get("category_hint")

    text = f"{title} {desc} {cond}"

    color = extract_color(text)
    category = extract_category(text) or cat_hint
    style_tags = extract_style_tags(text)
    condition_norm = normalize_condition(text)

    return {
        "color": color,
        "category": category,
        "style_tags": style_tags,
        "condition_norm": condition_norm,
    }


def apply_attribute_reasoner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the attribute reasoner to an entire DataFrame and
    return a new DataFrame with extra columns:
      - color
      - category
      - style_tags
      - condition_norm
    """
    attr_records = df.apply(infer_attributes_from_row, axis=1, result_type="expand")
    return pd.concat([df.reset_index(drop=True), attr_records], axis=1)
