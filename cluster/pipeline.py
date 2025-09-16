from __future__ import annotations

import json
import warnings
import re

import numpy as np
import pandas as pd
import hdbscan

try:
    from .embedding import encode_texts
except ImportError:  # pragma: no cover - fallback for direct execution
    from embedding import encode_texts  # type: ignore

# --- domain dictionaries you can extend ---
BRANDS = {
    "betway": "betway", "sportybet": "sportybet", "msport": "msport",
    "betpawa": "betpawa", "betika": "betika"
}
MODIFIERS = [
    "app","apk","login","register","bonus","odds","fixtures","live",
    "tips","predictions","jackpot","casino","slots","live dealer",
    "cashout","bet builder"
]
REGIONS = ["ghana","south africa","botswana","zambia","tanzania","mozambique"]

def normalize_kw(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-&]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for k, v in BRANDS.items():
        s = re.sub(rf"\b{k}\b", v, s)
    return s

def extract_tags(s: str):
    toks = s.split()
    brands = [b for b in BRANDS if b in toks]
    regions = [r for r in REGIONS if r in s]
    modifiers = [m for m in MODIFIERS if re.search(rf"\b{re.escape(m)}\b", s)]
    return brands, regions, modifiers

def cluster_keywords(
    keywords: list[str],
    *,
    min_cluster_size: int = 3,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
    model_path: str | None = None,
) -> pd.DataFrame:
    """Cluster keywords using embeddings and HDBSCAN.

    Parameters
    ----------
    keywords:
        The normalized keywords that should be clustered.
    min_cluster_size:
        Minimum size HDBSCAN should consider a cluster.  Defaults to 3 which
        keeps noise under control without forcing overly large clusters.
    min_samples:
        Optional density parameter forwarded to :class:`hdbscan.HDBSCAN`.
    cluster_selection_epsilon:
        Softens the cluster boundaries when greater than zero.  The default of 0
        mirrors the deterministic behaviour used previously.
    model_path:
        Optional override for the encoder path.
    """

    if not keywords:
        return pd.DataFrame(
            columns=["cluster_id", "keyword_norm", "centroid", "avg_sim", "cluster_confidence"]
        )

    embeddings = encode_texts(keywords, model_path=model_path)
    clusterer = hdbscan.HDBSCAN(
        metric="euclidean",
        min_cluster_size=max(2, min_cluster_size),
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    labels = clusterer.fit_predict(embeddings)

    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)

    rows = []
    next_cluster_id = 0

    def _emit_cluster(idxs: list[int], cluster_id: int) -> None:
        sub_emb = embeddings[idxs]
        sims = np.matmul(sub_emb, sub_emb.T)
        mean_sims = sims.mean(axis=1)
        centroid_idx = idxs[int(np.argmax(mean_sims))]
        centroid = keywords[centroid_idx]
        for pos, idx in enumerate(idxs):
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "keyword_norm": keywords[idx],
                    "centroid": centroid,
                    "avg_sim": float(mean_sims[pos]),
                    "cluster_confidence": float(clusterer.probabilities_[idx]),
                }
            )

    for label in sorted(k for k in clusters if k != -1):
        idxs = clusters[label]
        _emit_cluster(idxs, next_cluster_id)
        next_cluster_id += 1

    for idx in clusters.get(-1, []):
        rows.append(
            {
                "cluster_id": next_cluster_id,
                "keyword_norm": keywords[idx],
                "centroid": keywords[idx],
                "avg_sim": 1.0,
                "cluster_confidence": float(clusterer.probabilities_[idx]),
            }
        )
        next_cluster_id += 1

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

def run_pipeline(
    csv_in,
    csv_out,
    min_cluster_size=3,
    min_samples=None,
    cluster_selection_epsilon=0.0,
    config_path=None,
    encoder_path=None,
    min_sim=None,
):
    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)
        BRANDS.update(cfg.get("BRANDS", {}))
        MODIFIERS.extend([m for m in cfg.get("MODIFIERS", []) if m not in MODIFIERS])
        REGIONS.extend([r for r in cfg.get("REGIONS", []) if r not in REGIONS])

    df = pd.read_csv(csv_in)
    if "keyword" not in df.columns:
        raise ValueError("Input CSV must have a 'keyword' column.")
    df["keyword_norm"] = df["keyword"].apply(normalize_kw)
    tags = df["keyword_norm"].apply(extract_tags)
    df[["brands","regions","modifiers"]] = pd.DataFrame(tags.tolist(), index=df.index)
    if min_sim is not None:
        warnings.warn(
            "'min_sim' is deprecated; use 'min_cluster_size' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        approx_size = max(2, int(round(1.0 / max(1.0 - min_sim, 1e-3))))
        min_cluster_size = max(min_cluster_size, approx_size)

    cl = cluster_keywords(
        df["keyword_norm"].tolist(),
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        model_path=encoder_path,
    )
    df = df.merge(cl, on="keyword_norm", how="left")
    intents = df["keyword_norm"].apply(classify_intent)
    df["intent"] = intents.apply(lambda x: x[0])
    df["intent_conf"] = intents.apply(lambda x: x[1])
    df.to_csv(csv_out, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Keyword clustering and intent classification")
    parser.add_argument("csv_in")
    parser.add_argument("csv_out")
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--cluster-epsilon", type=float, default=0.0)
    parser.add_argument("--encoder-path")
    parser.add_argument("--min-sim", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--config", dest="config_path")
    args = parser.parse_args()
    run_pipeline(
        args.csv_in,
        args.csv_out,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_epsilon,
        config_path=args.config_path,
        encoder_path=args.encoder_path,
        min_sim=args.min_sim,
    )
