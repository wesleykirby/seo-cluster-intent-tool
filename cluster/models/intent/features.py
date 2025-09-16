"""Feature utilities for intent models."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ...text_utils import INTENT_RULES, normalize_kw


@dataclass
class IntentFeatureBuilder:
    """Build features from text, rule hits and SERP signals."""

    intent_rules: Sequence[Tuple[str, str]] = field(default_factory=lambda: INTENT_RULES)
    serp_prefix: str = "serp_"
    vectorizer: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    )

    def __post_init__(self) -> None:
        self._compiled_rules = [
            (label, re.compile(pattern, re.IGNORECASE)) for label, pattern in self.intent_rules
        ]
        self.serp_columns_: list[str] = []

    # Methods -----------------------------------------------------------------
    def fit(self, df) -> "IntentFeatureBuilder":
        texts = self._get_text_series(df)
        self.vectorizer.fit(texts)
        self.serp_columns_ = sorted(
            [col for col in df.columns if col.startswith(self.serp_prefix)]
        )
        return self

    def transform(self, df):
        texts = self._get_text_series(df)
        text_matrix = self.vectorizer.transform(texts)

        rule_features = np.zeros((len(df), len(self._compiled_rules)), dtype=float)
        for idx, (_, pattern) in enumerate(self._compiled_rules):
            rule_features[:, idx] = [1.0 if pattern.search(text) else 0.0 for text in texts]
        rule_matrix = sparse.csr_matrix(rule_features)

        if self.serp_columns_:
            serp_values = []
            for col in self.serp_columns_:
                if col in df.columns:
                    values = df[col].fillna(0).astype(float).to_numpy()
                else:
                    values = np.zeros(len(df), dtype=float)
                serp_values.append(values)
            serp_array = np.column_stack(serp_values) if serp_values else np.empty((len(df), 0))
            serp_matrix = sparse.csr_matrix(serp_array)
        else:
            serp_matrix = sparse.csr_matrix((len(df), 0))

        return sparse.hstack([text_matrix, rule_matrix, serp_matrix], format="csr")

    # Helpers -----------------------------------------------------------------
    def _get_text_series(self, df):
        if "keyword_norm" in df.columns:
            texts = df["keyword_norm"].fillna(df.get("keyword", ""))
        elif "keyword" in df.columns:
            texts = df["keyword"].fillna("")
        else:
            raise KeyError("DataFrame must contain either 'keyword' or 'keyword_norm' column")
        texts = [normalize_kw(str(text)) for text in texts]
        return texts


def get_text_series(df) -> list[str]:
    """Expose text normalisation helper for other inference utilities."""
    builder = IntentFeatureBuilder()
    return builder._get_text_series(df)
