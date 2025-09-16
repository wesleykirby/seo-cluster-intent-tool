"""Intent model abstractions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from .features import IntentFeatureBuilder, get_text_series
from ...text_utils import INTENT_RULES, classify_with_rules


@dataclass
class IntentModel:
    """Wrapper combining the feature builder and downstream classifier."""

    feature_builder: IntentFeatureBuilder
    classifier: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, df) -> np.ndarray:
        features = self.feature_builder.transform(df)
        if not hasattr(self.classifier, "predict_proba"):
            raise AttributeError("Classifier does not implement predict_proba")
        return self.classifier.predict_proba(features)

    def predict_with_prob(self, df) -> Tuple[np.ndarray, np.ndarray]:
        proba = self.predict_proba(df)
        classes = np.asarray(self.classifier.classes_)
        best_idx = np.argmax(proba, axis=1)
        best_labels = classes[best_idx]
        best_scores = proba[np.arange(proba.shape[0]), best_idx]
        return best_labels, best_scores

    @classmethod
    def from_serialized(cls, artifact: Any) -> "IntentModel":
        if isinstance(artifact, cls):
            return artifact
        if isinstance(artifact, dict):
            return cls(
                feature_builder=artifact["feature_builder"],
                classifier=artifact["classifier"],
                metadata=artifact.get("metadata", {}),
            )
        raise TypeError(f"Unsupported intent model artifact type: {type(artifact)!r}")


@dataclass
class RuleBasedIntentModel:
    """Fallback model that leverages rule hits only."""

    intent_rules: Iterable[Tuple[str, str]] = field(default_factory=lambda: INTENT_RULES)
    default_label: str = "unsure"
    base_confidence: float = 0.4

    def predict_with_prob(self, df) -> Tuple[np.ndarray, np.ndarray]:
        texts = get_text_series(df)
        labels = []
        probs = []
        for text in texts:
            label, conf = classify_with_rules(
                text,
                intent_rules=self.intent_rules,
                default_label=self.default_label,
                base_confidence=self.base_confidence,
            )
            labels.append(label)
            probs.append(conf)
        return np.array(labels), np.array(probs)
