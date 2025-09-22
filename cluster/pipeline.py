"""Keyword clustering and intent classification pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models.intent import load_intent_model
from .text_utils import apply_domain_config, extract_tags, normalize_kw, detect_industry_vertical
from .semantic_analyzer import SemanticKeywordAnalyzer


def cluster_keywords(keywords: list[str], min_sim: float = 0.8) -> pd.DataFrame:
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    matrix = vec.fit_transform(keywords)
    sim = cosine_similarity(matrix)
    n = len(keywords)
    visited, clusters = set(), []
    for i in range(n):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, n):
            if j in visited:
                continue
            if sim[i, j] >= min_sim:
                group.append(j)
                visited.add(j)
        clusters.append(group)

    rows = []
    for cid, idxs in enumerate(clusters):
        centroid_idx = max(
            idxs,
            key=lambda i: float(sum(sim[i][idxs]) / len(idxs)) if len(idxs) else 0.0,
        )
        centroid = keywords[centroid_idx]
        for i in idxs:
            rows.append(
                {
                    "cluster_id": cid,
                    "keyword_norm": keywords[i],
                    "centroid": centroid,
                    "avg_sim": float(sum(sim[i][idxs]) / len(idxs)),
                }
            )
    return pd.DataFrame(rows)


def run_pipeline(
    csv_in: str | Path,
    csv_out: str | Path,
    min_sim: float = 0.8,
    config_path: Optional[str | Path] = None,
    models_dir: Optional[str | Path] = None,
) -> None:
    """New semantic analysis pipeline that outputs Main/Sub/Modifier/Keyword structure"""
    
    df = pd.read_csv(csv_in)
    if "keyword" not in df.columns:
        raise ValueError("Input CSV must have a 'keyword' column.")
    
    print(f"🔍 Analyzing {len(df)} keywords with semantic ML...")
    
    # Use the new semantic analyzer
    analyzer = SemanticKeywordAnalyzer()
    results_df = analyzer.analyze_keywords(df["keyword"].tolist())
    
    print(f"✅ Analysis complete! Generated 4-column structure:")
    print(f"   - Discovered {len(analyzer.brand_patterns)} brands automatically")
    print(f"   - Classified into semantic topics and modifiers")
    
    # Save the clean 4-column output
    results_df.to_csv(csv_out, index=False)
    
    print(f"💾 Results saved to {csv_out}")


# Legacy function for backward compatibility
def run_pipeline_legacy(
    csv_in: str | Path,
    csv_out: str | Path,
    min_sim: float = 0.8,
    config_path: Optional[str | Path] = None,
    models_dir: Optional[str | Path] = None,
) -> None:
    """Legacy pipeline with old clustering approach - kept for reference"""
    if config_path:
        with open(config_path) as fh:
            cfg = json.load(fh)
        apply_domain_config(cfg)

    df = pd.read_csv(csv_in)
    if "keyword" not in df.columns:
        raise ValueError("Input CSV must have a 'keyword' column.")
    
    # Auto-detect vertical if not provided
    if "vertical" not in df.columns:
        detected_vertical = detect_industry_vertical(df["keyword"].tolist())
        df["vertical"] = detected_vertical
        print(f"Auto-detected industry vertical: {detected_vertical}")
    else:
        df["vertical"] = df["vertical"].fillna("general").astype(str)

    df["keyword_norm"] = df["keyword"].astype(str).apply(normalize_kw)
    tags = df["keyword_norm"].apply(extract_tags)
    df[["brands", "regions", "modifiers"]] = pd.DataFrame(tags.tolist(), index=df.index)

    clustering = cluster_keywords(df["keyword_norm"].tolist(), min_sim=min_sim)
    df = df.merge(clustering, on="keyword_norm", how="left", suffixes=("", "_cluster"))

    intents = pd.Series(index=df.index, dtype="object")
    probs = pd.Series(index=df.index, dtype=float)
    
    # Import rule-based classification as fallback
    from .text_utils import classify_with_rules
    
    for vertical, idxs in df.groupby("vertical").groups.items():
        try:
            # Ensure vertical is a string and models_dir is Path or None
            vertical_str = str(vertical)
            models_path = Path(models_dir) if models_dir else None
            model = load_intent_model(vertical_str, models_dir=models_path)
            subset = df.loc[idxs]
            labels, scores = model.predict_with_prob(subset)
            intents.loc[idxs] = labels
            probs.loc[idxs] = scores
        except (FileNotFoundError, ValueError, ImportError):
            # Fallback to rule-based classification if model doesn't exist
            print(f"Model for vertical '{vertical}' not found, using rule-based classification")
            subset = df.loc[idxs]
            rule_results = subset["keyword_norm"].apply(lambda x: classify_with_rules(x))
            intents.loc[idxs] = [result[0] for result in rule_results]
            probs.loc[idxs] = [result[1] for result in rule_results]

    df["intent"] = intents
    df["intent_prob"] = probs
    df["intent_conf"] = probs

    df.to_csv(csv_out, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Keyword clustering and intent classification")
    parser.add_argument("csv_in")
    parser.add_argument("csv_out")
    parser.add_argument("--min-sim", type=float, default=0.8)
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--models-dir", dest="models_dir")
    args = parser.parse_args()

    run_pipeline(
        args.csv_in,
        args.csv_out,
        min_sim=args.min_sim,
        config_path=args.config_path,
        models_dir=args.models_dir,
    )
