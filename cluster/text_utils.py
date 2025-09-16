"""Utility helpers for keyword normalization and intent rules."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

BRANDS: Dict[str, str] = {
    "betway": "betway",
    "sportybet": "sportybet",
    "msport": "msport",
    "betpawa": "betpawa",
    "betika": "betika",
}

MODIFIERS: List[str] = [
    "app",
    "apk",
    "login",
    "register",
    "bonus",
    "odds",
    "fixtures",
    "live",
    "tips",
    "predictions",
    "jackpot",
    "casino",
    "slots",
    "live dealer",
    "cashout",
    "bet builder",
]

REGIONS: List[str] = [
    "ghana",
    "south africa",
    "botswana",
    "zambia",
    "tanzania",
    "mozambique",
]

INTENT_RULES: List[Tuple[str, str]] = [
    ("informational", r"\b(how|what|why|guide|meaning|rules|strategy|tips|predictions)\b"),
    ("transactional", r"\b(register|sign up|login|download|app|apk|deposit|withdraw)\b"),
    ("commercial", r"\b(best|top|bonus|promo|odds|compare|vs|review)\b"),
    ("navigational", r"\b(betway|sportybet|msport|betpawa|site|website)\b"),
    ("local", r"\b(near me|ghana|south africa|botswana|zambia|tanzania|mozambique)\b"),
]

_ORDERED_INTENTS: Sequence[str] = (
    "transactional",
    "commercial",
    "informational",
    "navigational",
    "local",
)


def apply_domain_config(config: Dict[str, Iterable[str]]) -> None:
    """Update global domain dictionaries based on a configuration mapping."""
    global BRANDS, MODIFIERS, REGIONS
    if "BRANDS" in config:
        BRANDS.update({k: v for k, v in config["BRANDS"].items()})
    if "MODIFIERS" in config:
        new_mods = [m for m in config["MODIFIERS"] if m not in MODIFIERS]
        MODIFIERS.extend(new_mods)
    if "REGIONS" in config:
        new_regs = [r for r in config["REGIONS"] if r not in REGIONS]
        REGIONS.extend(new_regs)


def normalize_kw(text: str) -> str:
    """Normalize a keyword string by removing punctuation and harmonising brands."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-&]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for brand, replacement in BRANDS.items():
        text = re.sub(rf"\b{brand}\b", replacement, text)
    return text


def extract_tags(text: str) -> Tuple[List[str], List[str], List[str]]:
    tokens = text.split()
    brands = [brand for brand in BRANDS if brand in tokens]
    regions = [region for region in REGIONS if region in text]
    modifiers = [
        modifier
        for modifier in MODIFIERS
        if re.search(rf"\b{re.escape(modifier)}\b", text)
    ]
    return brands, regions, modifiers


def compute_rule_hits(text: str, intent_rules: Sequence[Tuple[str, str]] | None = None) -> Dict[str, int]:
    """Return counts of rule matches for a piece of text."""
    rules = intent_rules or INTENT_RULES
    scores: Dict[str, int] = {}
    for label, pattern in rules:
        if re.search(pattern, text):
            scores[label] = scores.get(label, 0) + 1
    return scores


def classify_with_rules(
    text: str,
    intent_rules: Sequence[Tuple[str, str]] | None = None,
    default_label: str = "unsure",
    base_confidence: float = 0.4,
) -> Tuple[str, float]:
    """Classify using regex rules and return the chosen label with pseudo confidence."""
    scores = compute_rule_hits(text, intent_rules=intent_rules)
    if not scores:
        return default_label, base_confidence

    intent_order = {intent: idx for idx, intent in enumerate(_ORDERED_INTENTS)}
    label = max(
        scores,
        key=lambda item: (
            scores[item],
            -intent_order.get(item, len(intent_order)),
        ),
    )
    confidence = min(0.95, 0.5 + 0.1 * len(scores))
    return label, confidence


def ensure_keyword_norm(series) -> List[str]:
    """Return a normalised keyword series, computing it if missing."""
    if series is None:
        raise ValueError("Keyword series cannot be None")
    return [normalize_kw(str(value)) for value in series]
