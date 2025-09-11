import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def run_pipeline(csv_in, csv_out):
    df = pd.read_csv(csv_in)
    if "keyword" not in df.columns:
        raise ValueError("Input CSV must have a 'keyword' column.")
    df["keyword_norm"] = df["keyword"].apply(normalize_kw)
    tags = df["keyword_norm"].apply(extract_tags)
    df[["brands","regions","modifiers"]] = pd.DataFrame(tags.tolist(), index=df.index)
    cl = cluster_keywords(df["keyword_norm"].tolist(), min_sim=0.8)
    df = df.merge(cl, on="keyword_norm", how="left")
    intents = df["keyword_norm"].apply(classify_intent)
    df["intent"] = intents.apply(lambda x: x[0])
    df["intent_conf"] = intents.apply(lambda x: x[1])
    df.to_csv(csv_out, index=False)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m cluster.pipeline <keywords_in.csv> <keywords_tagged.csv>")
        raise SystemExit(2)
    run_pipeline(sys.argv[1], sys.argv[2])
