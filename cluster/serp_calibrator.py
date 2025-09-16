"""SERP-driven calibration for intent probabilities.

This module exposes a :class:`SERPCalibrator` that fits a logistic regression
model on coarse SERP level flags (brand presence, register/login intents, etc).
It is designed to complement the heuristic model in :mod:`cluster.pipeline`
by providing an additional probability based on observed SERP results.

The calibrator operates on rows that optionally contain SERP metadata such as
``serp_urls``, ``serp_titles`` or ``serp_snippets``. The metadata can either be
stored as Python lists or as strings containing JSON/pipe/newline separated
values. Only lightweight flags are extracted so the calibrator can be trained
and evaluated even with sparse SERP information.

A lightweight bootstrap model is provided so that the pipeline can still
produce reasonable outputs even when no pre-trained model is available on disk.
For higher fidelity performance a dedicated training run can be launched via
``train_serp_calibrator.py`` which persists a calibrated model to disk.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

LOGGER = logging.getLogger(__name__)

SERP_URL_KEYS = [
    "serp_urls",
    "top_urls",
    "urls",
    "url_list",
    "top_links",
]
SERP_TITLE_KEYS = [
    "serp_titles",
    "top_titles",
    "titles",
]
SERP_SNIPPET_KEYS = [
    "serp_snippets",
    "top_snippets",
    "snippets",
    "descriptions",
]

DEFAULT_FEATURE_NAMES: List[str] = [
    "flag_register",
    "flag_login",
    "flag_app",
    "flag_bonus",
    "flag_money",
    "flag_brand_top1",
    "brand_presence_ratio",
    "flag_brand_anywhere",
    "flag_review",
    "flag_informational",
]

DEFAULT_MODEL_FILENAME = "serp_calibrator.joblib"

REGISTER_PATTERNS = ["/register", " register", "sign up", "create account"]
LOGIN_PATTERNS = ["/login", " login", "log in", "account"]
APP_PATTERNS = [" app", " apk", "download", "install"]
BONUS_PATTERNS = ["bonus", "promo", "offer", "freebet", "free bet"]
MONEY_PATTERNS = ["deposit", "withdraw", "cashout", "cash out", "payout"]
REVIEW_PATTERNS = ["review", "vs", "comparison", "best", "top"]
INFORMATIONAL_PATTERNS = ["how", "what", "why", "meaning", "guide", "tips"]


def _normalise_brands(brands: Optional[Iterable[str]]) -> List[str]:
    if not brands:
        return []
    seen = []
    for b in brands:
        if not b:
            continue
        norm = str(b).strip().lower()
        if norm and norm not in seen:
            seen.append(norm)
    return seen


def _parse_list_field(value) -> List[str]:
    """Parse a value that may represent a list of SERP entries."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # Attempt JSON decoding first.
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(v) for v in parsed if str(v).strip()]
        # Fall back to splitting on pipes/semicolons/newlines/commas.
        parts = re.split(r"\s*\|\s*|\s*;\s*|\s*\n\s*|\s*,\s*", text)
        return [p for p in parts if p]
    # Default fallback: wrap single primitive.
    return [str(value)]


def _extract_first_available(row: Mapping[str, object], keys: Sequence[str]) -> List[str]:
    for key in keys:
        if isinstance(row, pd.Series):
            if key not in row or pd.isna(row[key]):
                continue
            value = row[key]
        else:
            if key not in row:
                continue
            value = row[key]
        parsed = _parse_list_field(value)
        if parsed:
            return parsed
    return []


def _contains_pattern(text: str, patterns: Sequence[str]) -> bool:
    text = text.lower()
    return any(pat in text for pat in patterns)


def _brand_metrics(urls: Sequence[str], brands: Sequence[str]):
    """Return (brand_anywhere, brand_top1, brand_ratio_top3)."""
    if not urls or not brands:
        return False, False, 0.0
    brand_anywhere = False
    brand_top1 = False
    brand_hits_top3 = 0
    limit = min(3, len(urls))
    for idx, url in enumerate(urls):
        url_lower = url.lower()
        for brand in brands:
            if brand and brand in url_lower:
                brand_anywhere = True
                if idx == 0:
                    brand_top1 = True
                if idx < 3:
                    brand_hits_top3 += 1
                break
    if limit == 0:
        ratio = 0.0
    else:
        ratio = brand_hits_top3 / float(limit)
    return brand_anywhere, brand_top1, ratio


def extract_serp_features(
    row: Mapping[str, object],
    brands: Optional[Iterable[str]] = None,
    feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
) -> np.ndarray:
    """Extract SERP feature vector for a single row."""
    brand_list = _normalise_brands(brands)
    urls = _extract_first_available(row, SERP_URL_KEYS)
    titles = _extract_first_available(row, SERP_TITLE_KEYS)
    snippets = _extract_first_available(row, SERP_SNIPPET_KEYS)

    combined_text_parts = []
    if titles:
        combined_text_parts.extend(titles)
    if snippets:
        combined_text_parts.extend(snippets)
    combined_text = " ".join(combined_text_parts).lower()
    url_text = " ".join(urls).lower()

    flag_register = int(
        _contains_pattern(url_text, REGISTER_PATTERNS)
        or _contains_pattern(combined_text, REGISTER_PATTERNS)
    )
    flag_login = int(
        _contains_pattern(url_text, LOGIN_PATTERNS)
        or _contains_pattern(combined_text, LOGIN_PATTERNS)
    )
    flag_app = int(
        _contains_pattern(url_text, APP_PATTERNS)
        or _contains_pattern(combined_text, APP_PATTERNS)
    )
    flag_bonus = int(
        _contains_pattern(url_text, BONUS_PATTERNS)
        or _contains_pattern(combined_text, BONUS_PATTERNS)
    )
    flag_money = int(
        _contains_pattern(url_text, MONEY_PATTERNS)
        or _contains_pattern(combined_text, MONEY_PATTERNS)
    )
    flag_review = int(_contains_pattern(combined_text, REVIEW_PATTERNS))
    flag_info = int(_contains_pattern(combined_text, INFORMATIONAL_PATTERNS))

    brand_anywhere, brand_top1, brand_ratio = _brand_metrics(urls, brand_list)

    feature_map = {
        "flag_register": float(flag_register),
        "flag_login": float(flag_login),
        "flag_app": float(flag_app),
        "flag_bonus": float(flag_bonus),
        "flag_money": float(flag_money),
        "flag_brand_top1": float(int(brand_top1)),
        "brand_presence_ratio": float(brand_ratio),
        "flag_brand_anywhere": float(int(brand_anywhere)),
        "flag_review": float(flag_review),
        "flag_informational": float(flag_info),
    }

    return np.array([feature_map.get(name, 0.0) for name in feature_names], dtype=float)


@dataclass
class SERPCalibrator:
    """Logistic regression calibrator based on SERP flags."""

    brands: Optional[Iterable[str]] = None
    feature_names: Sequence[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_NAMES))
    model: Optional[LogisticRegression] = None

    def __post_init__(self) -> None:
        self.brands = _normalise_brands(self.brands)
        if self.model is None:
            self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
        self._is_bootstrap = False

    # -- feature helpers -----------------------------------------------------------------
    def transform_row(self, row: Mapping[str, object]) -> np.ndarray:
        return extract_serp_features(row, brands=self.brands, feature_names=self.feature_names)

    def transform_frame(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty:
            return np.zeros((0, len(self.feature_names)))
        matrix = np.vstack([self.transform_row(row) for _, row in df.iterrows()])
        return matrix

    # -- training ------------------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        label_col: str,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "SERPCalibrator":
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in dataframe")
        X = self.transform_frame(df)
        y = df[label_col].astype(int).to_numpy()
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
        self.model.fit(X, y, sample_weight=sample_weight)
        self._is_bootstrap = False
        return self

    def _fit_from_feature_matrix(
        self, X: np.ndarray, y: Sequence[int], sample_weight: Optional[Sequence[float]] = None
    ) -> "SERPCalibrator":
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    # -- inference -----------------------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        check_is_fitted(self.model)
        X = self.transform_frame(df)
        if X.size == 0:
            return pd.Series([], index=df.index, dtype=float)
        probs = self.model.predict_proba(X)[:, 1]
        return pd.Series(probs, index=df.index)

    def predict_row(self, row: Mapping[str, object]) -> float:
        check_is_fitted(self.model)
        vec = self.transform_row(row).reshape(1, -1)
        prob = float(self.model.predict_proba(vec)[0, 1])
        return prob

    # -- persistence ---------------------------------------------------------------------
    def save(self, path: os.PathLike) -> Path:
        check_is_fitted(self.model)
        payload = {
            "model": self.model,
            "feature_names": list(self.feature_names),
            "brands": list(self.brands),
            "is_bootstrap": getattr(self, "_is_bootstrap", False),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)
        LOGGER.info("Saved SERP calibrator to %s", path)
        return path

    @classmethod
    def load(cls, path: os.PathLike) -> "SERPCalibrator":
        data = joblib.load(path)
        model = data.get("model")
        calibrator = cls(
            brands=data.get("brands"),
            feature_names=data.get("feature_names", DEFAULT_FEATURE_NAMES),
            model=model,
        )
        calibrator._is_bootstrap = data.get("is_bootstrap", False)
        return calibrator

    # -- misc ----------------------------------------------------------------------------
    @property
    def is_bootstrap(self) -> bool:
        return getattr(self, "_is_bootstrap", False)


def _bootstrap_calibrator(brands: Optional[Iterable[str]] = None) -> SERPCalibrator:
    """Create a heuristic logistic regression when no persisted model exists."""
    calibrator = SERPCalibrator(brands=brands)
    # Synthetic feature matrix that encodes rough intuition about transactional SERPs.
    positives = np.array(
        [
            [1, 0, 0, 1, 1, 1, 1, 1, 0, 0],  # register + brand presence
            [1, 1, 1, 0, 1, 1, 0.67, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0.5, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0.4, 1, 0, 0],
            [1, 0, 1, 0, 1, 1, 0.8, 1, 0, 0],
        ],
        dtype=float,
    )
    negatives = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0.0, 0, 1, 1],  # informational guides
            [0, 0, 0, 0, 0, 0, 0.0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0.0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.0, 1, 1, 1],  # brand present but review heavy
            [0, 0, 0, 1, 0, 0, 0.0, 0, 1, 1],
        ],
        dtype=float,
    )
    X = np.vstack([positives, negatives])
    y = np.array([1] * len(positives) + [0] * len(negatives))
    calibrator._fit_from_feature_matrix(X, y)
    calibrator._is_bootstrap = True
    LOGGER.warning(
        "No trained SERP calibrator found. Falling back to heuristic bootstrap coefficients."
    )
    return calibrator


def load_default_calibrator(
    brands: Optional[Iterable[str]] = None,
    model_path: Optional[os.PathLike] = None,
) -> SERPCalibrator:
    """Load the persisted calibrator or fall back to the bootstrap version."""
    if model_path is None:
        model_path = Path(__file__).with_name(DEFAULT_MODEL_FILENAME)
    path = Path(model_path)
    if path.exists():
        try:
            calibrator = SERPCalibrator.load(path)
            if brands:
                calibrator.brands = _normalise_brands(brands)
            LOGGER.info("Loaded SERP calibrator from %s (bootstrap=%s)", path, calibrator.is_bootstrap)
            return calibrator
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Failed to load SERP calibrator at %s: %s", path, exc)
    return _bootstrap_calibrator(brands=brands)


__all__ = [
    "SERPCalibrator",
    "load_default_calibrator",
    "extract_serp_features",
    "DEFAULT_FEATURE_NAMES",
    "DEFAULT_MODEL_FILENAME",
]
