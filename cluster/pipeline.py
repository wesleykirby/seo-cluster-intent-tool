import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import hdbscan
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CONSTRAINTS_DIR = Path(__file__).resolve().parent
MUST_LINK_FILE = CONSTRAINTS_DIR / "must_link.json"
CANNOT_LINK_FILE = CONSTRAINTS_DIR / "cannot_link.json"

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


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

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

def load_constraint_pairs(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("pairs", [])
    pairs: List[Tuple[str, str]] = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            a = normalize_kw(str(item[0]))
            b = normalize_kw(str(item[1]))
            if a and b:
                pairs.append((a, b))
    # preserve order while removing duplicates
    seen = set()
    uniq_pairs = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            uniq_pairs.append(pair)
    return uniq_pairs

def load_constraints() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    return load_constraint_pairs(MUST_LINK_FILE), load_constraint_pairs(CANNOT_LINK_FILE)

def _hdbscan_min_cluster_size(n_items: int, min_sim: float) -> int:
    if n_items <= 2:
        return max(1, n_items)
    span = max(2, int(round((1 - min_sim) * n_items)))
    return min(max(span, 2), n_items)

def _prepare_initial_labels(raw_labels: np.ndarray) -> np.ndarray:
    labels = np.array(raw_labels, dtype=int)
    if labels.size == 0:
        return labels
    positives = labels[labels >= 0]
    next_label = int(positives.max() + 1) if positives.size else 0
    for idx, value in enumerate(labels):
        if value == -1:
            labels[idx] = next_label
            next_label += 1
    return labels

def _build_index_lookup(keywords: Sequence[str]) -> Dict[str, List[int]]:
    lookup: Dict[str, List[int]] = defaultdict(list)
    for idx, kw in enumerate(keywords):
        lookup[kw].append(idx)
    return lookup

def merge_must_links(labels: np.ndarray, lookup: Dict[str, List[int]], must_links: Sequence[Tuple[str, str]]) -> np.ndarray:
    if not must_links:
        return labels
    n = len(labels)
    uf = UnionFind(n)
    clusters: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if label >= 0:
            clusters[label].append(idx)
    for members in clusters.values():
        anchor = members[0]
        for idx in members[1:]:
            uf.union(anchor, idx)
    missing: Set[str] = set()
    for a, b in must_links:
        idxs_a = lookup.get(a, [])
        idxs_b = lookup.get(b, [])
        if not idxs_a:
            missing.add(a)
        if not idxs_b:
            missing.add(b)
        for ia in idxs_a:
            for ib in idxs_b:
                uf.union(ia, ib)
    new_labels = labels.copy()
    root_to_label: Dict[int, int] = {}
    next_label = 0
    for idx in range(n):
        root = uf.find(idx)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        new_labels[idx] = root_to_label[root]
    if missing:
        print(f"[cluster] Must-link keywords not found: {', '.join(sorted(missing))}")
    return new_labels

def _split_cluster(indices: List[int], conflicts: Sequence[Tuple[int, int]], features: np.ndarray) -> List[List[int]]:
    if len(indices) <= 1 or not conflicts:
        return [indices]
    index_set = set(indices)
    relevant = [pair for pair in conflicts if pair[0] in index_set and pair[1] in index_set]
    if len(relevant) <= 1 and len(indices) == 2:
        pair = relevant[0] if relevant else None
        if pair and set(pair) == set(indices):
            return [[indices[0]], [indices[1]]]
    if not relevant:
        return [indices]
    subset = features[indices]
    groups: List[List[int]] = []
    try:
        km = KMeans(n_clusters=2, random_state=0, n_init=10)
        local_labels = km.fit_predict(subset)
        unique_local = sorted(set(local_labels))
        for lbl in unique_local:
            group = [indices[i] for i, val in enumerate(local_labels) if val == lbl]
            if group:
                groups.append(group)
    except Exception:
        groups = []
    if len(groups) < 2:
        mid = len(indices) // 2
        if mid == 0:
            return [indices]
        groups = [indices[:mid], indices[mid:]]
    result: List[List[int]] = []
    for group in groups:
        if len(group) <= 1:
            result.append(group)
            continue
        sub_conflicts = [pair for pair in relevant if pair[0] in group and pair[1] in group]
        if sub_conflicts:
            result.extend(_split_cluster(group, sub_conflicts, features))
        else:
            result.append(group)
    return result

def split_cannot_links(labels: np.ndarray, lookup: Dict[str, List[int]], cannot_links: Sequence[Tuple[str, str]], features: np.ndarray) -> np.ndarray:
    if not cannot_links:
        return labels
    new_labels = labels.copy()
    missing: Set[str] = set()
    next_label = int(new_labels.max() + 1) if new_labels.size else 0
    while True:
        conflicts_by_cluster: Dict[int, Set[Tuple[int, int]]] = {}
        for a, b in cannot_links:
            idxs_a = lookup.get(a, [])
            idxs_b = lookup.get(b, [])
            if not idxs_a:
                missing.add(a)
            if not idxs_b:
                missing.add(b)
            for ia in idxs_a:
                for ib in idxs_b:
                    if ia == ib:
                        continue
                    if new_labels[ia] == new_labels[ib]:
                        cid = int(new_labels[ia])
                        pair = tuple(sorted((ia, ib)))
                        conflicts_by_cluster.setdefault(cid, set()).add(pair)
        if not conflicts_by_cluster:
            break
        changed = False
        for cid in sorted(conflicts_by_cluster):
            members = [idx for idx, lbl in enumerate(new_labels) if lbl == cid]
            if len(members) <= 1:
                continue
            splits = _split_cluster(members, list(conflicts_by_cluster[cid]), features)
            splits = [sorted(group) for group in splits if group]
            if len(splits) <= 1:
                continue
            splits.sort(key=lambda g: (min(g), len(g)))
            label_values = [cid] + list(range(next_label, next_label + len(splits) - 1))
            next_label += len(splits) - 1
            for group, value in zip(splits, label_values):
                for idx in group:
                    new_labels[idx] = value
            changed = True
        if not changed:
            break
    if missing:
        print(f"[cluster] Cannot-link keywords not found: {', '.join(sorted(missing))}")
    return new_labels

def adjust_with_constraints(labels: np.ndarray, keywords: Sequence[str], features: np.ndarray, must_links: Sequence[Tuple[str, str]], cannot_links: Sequence[Tuple[str, str]]) -> np.ndarray:
    lookup = _build_index_lookup(keywords)
    merged = merge_must_links(labels, lookup, must_links)
    adjusted = split_cannot_links(merged, lookup, cannot_links, features)
    return adjusted

def _reindex_labels(labels: Sequence[int]) -> np.ndarray:
    mapping: Dict[int, int] = {}
    next_label = 0
    reordered = np.empty(len(labels), dtype=int)
    for idx, label in enumerate(labels):
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
        reordered[idx] = mapping[label]
    return reordered

def _build_cluster_frame(keywords: Sequence[str], labels: Sequence[int], features: np.ndarray) -> pd.DataFrame:
    clusters: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    rows: List[Dict[str, object]] = []
    for cid in sorted(clusters):
        members = sorted(clusters[cid])
        cluster_features = features[members]
        if len(members) == 1:
            centroid_idx = members[0]
            avg_sims = [1.0]
        else:
            sims = cosine_similarity(cluster_features)
            avg_sims = sims.mean(axis=1)
            centroid_local = int(np.argmax(avg_sims))
            centroid_idx = members[centroid_local]
        centroid_kw = keywords[centroid_idx]
        for pos, idx in enumerate(members):
            avg_sim = float(avg_sims[pos]) if len(members) > 1 else 1.0
            rows.append({
                "cluster_id": cid,
                "keyword_norm": keywords[idx],
                "centroid": centroid_kw,
                "avg_sim": avg_sim,
            })
    return pd.DataFrame(rows)

def cluster_keywords(keywords: List[str], min_sim=0.8, must_links: Optional[Iterable[Tuple[str, str]]] = None, cannot_links: Optional[Iterable[Tuple[str, str]]] = None) -> pd.DataFrame:
    if not keywords:
        return pd.DataFrame(columns=["cluster_id", "keyword_norm", "centroid", "avg_sim"])
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5))
    X = vec.fit_transform(keywords)
    features = X.toarray()
    n_items = len(keywords)
    min_cluster_size = _hdbscan_min_cluster_size(n_items, min_sim)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(1, min_cluster_size // 2),
        metric="euclidean",
        allow_single_cluster=True,
    )
    raw_labels = clusterer.fit_predict(features)
    labels = _prepare_initial_labels(raw_labels)
    must_links_list = list(must_links or [])
    cannot_links_list = list(cannot_links or [])
    if must_links_list or cannot_links_list:
        labels = adjust_with_constraints(labels, keywords, features, must_links_list, cannot_links_list)
    labels = _reindex_labels(labels)
    return _build_cluster_frame(keywords, labels, features)

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

def run_pipeline(csv_in, csv_out, min_sim=0.8, config_path=None):
    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)
        BRANDS.update(cfg.get("BRANDS", {}))
        MODIFIERS.extend([m for m in cfg.get("MODIFIERS", []) if m not in MODIFIERS])
        REGIONS.extend([r for r in cfg.get("REGIONS", []) if r not in REGIONS])
    must_links, cannot_links = load_constraints()
    df = pd.read_csv(csv_in)
    if "keyword" not in df.columns:
        raise ValueError("Input CSV must have a 'keyword' column.")
    df["keyword_norm"] = df["keyword"].apply(normalize_kw)
    tags = df["keyword_norm"].apply(extract_tags)
    df[["brands","regions","modifiers"]] = pd.DataFrame(tags.tolist(), index=df.index)
    cl = cluster_keywords(df["keyword_norm"].tolist(), min_sim=min_sim, must_links=must_links, cannot_links=cannot_links)
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
    parser.add_argument("--min-sim", type=float, default=0.8)
    parser.add_argument("--config", dest="config_path")
    args = parser.parse_args()
    run_pipeline(args.csv_in, args.csv_out, min_sim=args.min_sim, config_path=args.config_path)
