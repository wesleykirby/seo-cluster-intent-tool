"""Model registry for loading vertical-specific intent models."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib

from .model import IntentModel, RuleBasedIntentModel
from ...text_utils import INTENT_RULES

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent
_MODEL_CACHE: Dict[Tuple[str, str], IntentModel] = {}


def _cache_key(vertical: str, models_dir: Path) -> Tuple[str, str]:
    return (str(models_dir), vertical)


def get_model_path(vertical: str, models_dir: Path | None = None) -> Path:
    base_dir = models_dir or _DEFAULT_MODEL_DIR
    return base_dir / f"{vertical}.joblib"


def load_intent_model(
    vertical: str,
    models_dir: Path | None = None,
    fallback_rules: Iterable[Tuple[str, str]] | None = None,
) -> RuleBasedIntentModel | IntentModel:
    base_dir = Path(models_dir) if models_dir else _DEFAULT_MODEL_DIR
    key = _cache_key(vertical, base_dir)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model_path = get_model_path(vertical, base_dir)
    if model_path.exists():
        artifact = joblib.load(model_path)
        model = IntentModel.from_serialized(artifact)
    else:
        model = RuleBasedIntentModel(intent_rules=fallback_rules or INTENT_RULES)
    _MODEL_CACHE[key] = model
    return model


def available_verticals(models_dir: Path | None = None) -> Dict[str, Path]:
    base_dir = Path(models_dir) if models_dir else _DEFAULT_MODEL_DIR
    mapping = {}
    if not base_dir.exists():
        return mapping
    for path in base_dir.glob("*.joblib"):
        mapping[path.stem] = path
    return mapping
