import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_LEXICON_PATH = Path(__file__).with_name("lexicon.json")


def load_lexicon(path: Path = DEFAULT_LEXICON_PATH) -> Tuple[Dict[str, str], List[str], List[str]]:
    """Load the brand/modifier/region lexicon from disk."""
    with open(path) as f:
        data = json.load(f)

    brands = {k.lower(): v.lower() for k, v in data.get("BRANDS", {}).items()}
    modifiers = [m.lower() for m in data.get("MODIFIERS", [])]
    regions = [r.lower() for r in data.get("REGIONS", [])]
    return brands, modifiers, regions


def merge_lexicon(
    brands: Dict[str, str],
    modifiers: Sequence[str],
    regions: Sequence[str],
    updates: Dict[str, Iterable],
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """Merge runtime updates with the base lexicon without creating duplicates."""

    merged_brands = dict(brands)
    for alias, canonical in updates.get("BRANDS", {}).items():
        alias_norm = str(alias).lower()
        canonical_norm = str(canonical).lower()
        merged_brands[alias_norm] = canonical_norm
        if canonical_norm not in merged_brands:
            merged_brands[canonical_norm] = canonical_norm

    merged_modifiers: List[str] = list(modifiers)
    for modifier in updates.get("MODIFIERS", []):
        mod_norm = str(modifier).lower()
        if mod_norm not in merged_modifiers:
            merged_modifiers.append(mod_norm)

    merged_regions: List[str] = list(regions)
    for region in updates.get("REGIONS", []):
        region_norm = str(region).lower()
        if region_norm not in merged_regions:
            merged_regions.append(region_norm)

    return merged_brands, merged_modifiers, merged_regions


def prepare_lexicon(config_path: str = None, lexicon_path: str = None):
    """Load the base lexicon and apply optional runtime configuration."""

    path = Path(lexicon_path) if lexicon_path else DEFAULT_LEXICON_PATH
    brands, modifiers, regions = load_lexicon(path)

    if config_path:
        with open(config_path) as f:
            updates = json.load(f)
        brands, modifiers, regions = merge_lexicon(brands, modifiers, regions, updates)

    return brands, modifiers, regions


BRANDS, MODIFIERS, REGIONS = load_lexicon()


def normalize_kw(s: str, brands: Dict[str, str] = None) -> str:
    brands = brands or BRANDS
    text = str(s).lower()
    text = re.sub(r"[^a-z0-9\s\-&]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for alias, canonical in brands.items():
        text = re.sub(rf"\b{re.escape(alias)}\b", canonical, text)
    return text


def extract_tags(
    s: str,
    brands: Dict[str, str] = None,
    regions: Iterable[str] = None,
    modifiers: Iterable[str] = None,
):
    brands = brands or BRANDS
    regions = list(regions or REGIONS)
    modifiers = list(modifiers or MODIFIERS)

    toks = str(s).split()
    brand_hits: List[str] = []
    for tok in toks:
        if tok in brands:
            canonical = brands[tok]
            if canonical not in brand_hits:
                brand_hits.append(canonical)

    text = str(s)
    region_hits: List[str] = []
    for region in regions:
        if region in text and region not in region_hits:
            region_hits.append(region)

    modifier_hits: List[str] = []
    for modifier in modifiers:
        if re.search(rf"\b{re.escape(modifier)}\b", text):
            modifier_hits.append(modifier)

    return brand_hits, region_hits, modifier_hits

def cluster_keywords(keywords: list, min_sim=0.8) -> pd.DataFrame:
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5))
    X = vec.fit_transform(keywords)
    sim = cosine_similarity(X)
    n = len(keywords)
    visited, clusters = set(), []
    for i in range(n):
        if i in visited: continue
        group = [i]; visited.add(i)
        for j in range(i+1, n):
            if j in visited: continue
            if sim[i, j] >= min_sim:
                group.append(j); visited.add(j)
        clusters.append(group)

    rows = []
    for cid, idxs in enumerate(clusters):
        centroid_idx = max(idxs, key=lambda i: float(sum(sim[i][idxs]) / len(idxs)))
        centroid = keywords[centroid_idx]
        for i in idxs:
            rows.append({
                "cluster_id": cid,
                "keyword_norm": keywords[i],
                "centroid": centroid,
                "avg_sim": float(sum(sim[i][idxs]) / len(idxs)),
            })
    return pd.DataFrame(rows)

INTENT_RULES = [
    ("informational", r"\b(how|what|why|guide|meaning|rules|strategy|tips|predictions)\b"),
    ("transactional", r"\b(register|sign up|login|download|app|apk|deposit|withdraw)\b"),
    ("commercial", r"\b(best|top|bonus|promo|odds|compare|vs|review)\b"),
    ("navigational", r"\b(betway|sportybet|msport|betpawa|site|website)\b"),
    ("local", r"\b(near me|ghana|south africa|botswana|zambia|tanzania|mozambique)\b"),
]

def classify_intent(text: str):
    scores = {}
    for label, pattern in INTENT_RULES:
        if re.search(pattern, text):
            scores[label] = scores.get(label, 0) + 1
    if not scores: return "unsure", 0.4
    order = ["transactional","commercial","informational","navigational","local"]
    label = max(scores, key=lambda k: (scores[k], -order.index(k) if k in order else 99))
    conf = min(0.9, 0.5 + 0.1*len(scores))
    return label, conf

def process_dataframe(
    df: pd.DataFrame,
    min_sim: float = 0.8,
    lexicon: Tuple[Dict[str, str], Sequence[str], Sequence[str]] = None,
) -> pd.DataFrame:
    """Run the clustering pipeline against an in-memory dataframe."""

    if "keyword" not in df.columns:
        raise ValueError("Input CSV must have a 'keyword' column.")

    brands, modifiers, regions = lexicon or (BRANDS, MODIFIERS, REGIONS)
    brands = dict(brands)
    modifiers = list(modifiers)
    regions = list(regions)

    result = df.copy()
    result["keyword_norm"] = result["keyword"].apply(lambda x: normalize_kw(x, brands=brands))
    tags = result["keyword_norm"].apply(
        lambda text: extract_tags(text, brands=brands, regions=regions, modifiers=modifiers)
    )
    result[["brands", "regions", "modifiers"]] = pd.DataFrame(tags.tolist(), index=result.index)

    clusters = cluster_keywords(result["keyword_norm"].tolist(), min_sim=min_sim)
    result = result.merge(clusters, on="keyword_norm", how="left")

    intents = result["keyword_norm"].apply(classify_intent)
    result["intent"] = intents.apply(lambda x: x[0])
    result["intent_conf"] = intents.apply(lambda x: x[1])

    return result


def run_pipeline(csv_in, csv_out, min_sim=0.8, config_path=None, lexicon_path=None):
    brands, modifiers, regions = prepare_lexicon(config_path=config_path, lexicon_path=lexicon_path)

    df = pd.read_csv(csv_in)
    result = process_dataframe(df, min_sim=min_sim, lexicon=(brands, modifiers, regions))
    result.to_csv(csv_out, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Keyword clustering and intent classification")
    parser.add_argument("csv_in")
    parser.add_argument("csv_out")
    parser.add_argument("--min-sim", type=float, default=0.8)
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--lexicon", dest="lexicon_path")
    args = parser.parse_args()
    run_pipeline(
        args.csv_in,
        args.csv_out,
        min_sim=args.min_sim,
        config_path=args.config_path,
        lexicon_path=args.lexicon_path,
    )
