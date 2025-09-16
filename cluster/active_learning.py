"""Utilities for maintaining a human-labeling queue for low confidence intents."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

DEFAULT_QUEUE_PATH = Path("data/label_queue.csv")
_QUEUE_PRIORITY = [
    "keyword",
    "keyword_norm",
    "cluster_id",
    "centroid",
    "intent",
    "intent_conf",
    "avg_sim",
    "brands",
    "regions",
    "modifiers",
    "human_intent",
]


def _ordered_columns(*dfs: pd.DataFrame) -> list[str]:
    """Return a stable column ordering prioritising the queue defaults."""
    seen = set()
    available: list[str] = []
    for df in dfs:
        for col in df.columns:
            if col not in available:
                available.append(col)
    ordered: list[str] = []
    for col in _QUEUE_PRIORITY:
        if col in available and col not in seen:
            ordered.append(col)
            seen.add(col)
    for col in available:
        if col not in seen:
            ordered.append(col)
            seen.add(col)
    if "human_intent" not in seen:
        ordered.append("human_intent")
    return ordered


def filter_low_confidence(
    df: pd.DataFrame,
    threshold: float = 0.6,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return rows whose ``intent_conf`` falls below ``threshold``.

    Parameters
    ----------
    df:
        The DataFrame output from the clustering pipeline. It must contain an
        ``intent_conf`` column.
    threshold:
        Minimum confidence required for a prediction to be accepted. Rows below
        this value will be returned.
    columns:
        Optional iterable restricting which columns are returned.
    """
    if "intent_conf" not in df.columns:
        raise ValueError("DataFrame must include an 'intent_conf' column")

    scores = pd.to_numeric(df["intent_conf"], errors="coerce").fillna(0.0)
    mask = scores < threshold
    candidates = df.loc[mask].copy()
    if candidates.empty:
        return candidates

    candidates["intent_conf"] = scores.loc[candidates.index]
    if "keyword" in candidates.columns:
        candidates = candidates.drop_duplicates(subset=["keyword"], keep="first")
    order = (
        list(columns)
        if columns is not None
        else _ordered_columns(candidates)
    )
    subset = [c for c in order if c in candidates.columns]
    result = candidates[subset].reset_index(drop=True)
    if "human_intent" not in result.columns:
        result["human_intent"] = ""
    return result


def load_label_queue(queue_path: Path | str = DEFAULT_QUEUE_PATH) -> pd.DataFrame:
    """Load the existing label queue from disk."""
    queue_path = Path(queue_path)
    if not queue_path.exists():
        columns = [c for c in _QUEUE_PRIORITY if c != "human_intent"] + ["human_intent"]
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(queue_path)
    if "human_intent" not in df.columns:
        df["human_intent"] = ""
    return df.fillna("")


def save_label_queue(df: pd.DataFrame, queue_path: Path | str = DEFAULT_QUEUE_PATH) -> None:
    """Persist the label queue to ``queue_path``."""
    queue_path = Path(queue_path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_cols = _ordered_columns(df)
    subset = [c for c in ordered_cols if c in df.columns]
    to_save = df.copy()[subset]
    if "human_intent" in to_save.columns:
        to_save["human_intent"] = to_save["human_intent"].fillna("")
    if "intent_conf" in to_save.columns:
        to_save["intent_conf"] = pd.to_numeric(
            to_save["intent_conf"], errors="coerce"
        )
    sort_cols: list[str] = []
    ascending: list[bool] = []
    if "human_intent" in to_save.columns:
        to_save["_has_label"] = to_save["human_intent"].astype(str).str.strip() != ""
        sort_cols.append("_has_label")
        ascending.append(True)
    if "intent_conf" in to_save.columns:
        sort_cols.append("intent_conf")
        ascending.append(True)
    if sort_cols:
        to_save = to_save.sort_values(sort_cols, ascending=ascending)
    if "_has_label" in to_save.columns:
        to_save = to_save.drop(columns=["_has_label"])
    to_save.fillna("", inplace=True)
    to_save.to_csv(queue_path, index=False)


def update_label_queue(
    df: Optional[pd.DataFrame] = None,
    *,
    candidates: Optional[pd.DataFrame] = None,
    threshold: float = 0.6,
    queue_path: Path | str = DEFAULT_QUEUE_PATH,
) -> pd.DataFrame:
    """Update the on-disk label queue and return the merged queue DataFrame.

    Either ``df`` or pre-filtered ``candidates`` must be provided. The function
    preserves any existing manual labels while refreshing low-confidence items.
    """
    if candidates is None:
        if df is None:
            raise ValueError("Either 'df' or 'candidates' must be provided")
        candidates = filter_low_confidence(df, threshold=threshold)
    else:
        candidates = candidates.copy()

    if "human_intent" not in candidates.columns:
        candidates["human_intent"] = ""

    existing = load_label_queue(queue_path)
    if not existing.empty and "human_intent" not in existing.columns:
        existing["human_intent"] = ""

    if existing.empty:
        merged_frames = [candidates] if not candidates.empty else []
    else:
        has_label = existing["human_intent"].astype(str).str.strip() != ""
        labeled = existing.loc[has_label]
        unlabeled = existing.loc[~has_label]
        if not candidates.empty:
            candidate_keywords = set(candidates.get("keyword", []))
            if "keyword" in unlabeled.columns and candidate_keywords:
                unlabeled = unlabeled[~unlabeled["keyword"].isin(candidate_keywords)]
            if "keyword" in labeled.columns and candidate_keywords:
                candidates = candidates[~candidates["keyword"].isin(labeled["keyword"])]
        merged_frames = [frame for frame in (candidates, unlabeled, labeled) if not frame.empty]

    if merged_frames:
        combined = pd.concat(merged_frames, ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame(columns=_ordered_columns(candidates, existing))

    ordered_cols = _ordered_columns(combined)
    combined = combined.reindex(columns=ordered_cols, fill_value="")
    save_label_queue(combined, queue_path=queue_path)
    return combined
