import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .pipeline import prepare_lexicon, process_dataframe


def _collect_novel_candidates(
    df: pd.DataFrame,
    brands: Dict[str, str],
    modifiers: Sequence[str],
    regions: Sequence[str],
    min_token_count: int = 1,
    min_phrase_count: int = 2,
) -> Tuple[List[Dict], List[Dict]]:
    """Derive token and phrase candidates that are not in the current lexicon."""

    brand_vocab = {b.lower() for b in brands.keys()} | {b.lower() for b in brands.values()}
    modifier_vocab = {m.lower() for m in modifiers}
    region_vocab = {r.lower() for r in regions}
    known_phrases = {m for m in modifier_vocab if " " in m}

    token_counts: Counter = Counter()
    token_examples: Dict[str, set] = defaultdict(set)
    phrase_counts: Counter = Counter()
    phrase_examples: Dict[str, set] = defaultdict(set)

    for original, normalized in zip(df["keyword"], df["keyword_norm"]):
        tokens = [tok for tok in str(normalized).split() if tok]
        for tok in tokens:
            if not re.search(r"[a-z]", tok):
                continue
            token_counts[tok] += 1
            if len(token_examples[tok]) < 5:
                token_examples[tok].add(str(original))
        for n in (2, 3):
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i : i + n])
                phrase_counts[phrase] += 1
                if len(phrase_examples[phrase]) < 5:
                    phrase_examples[phrase].add(str(original))

    novel_tokens: List[Dict] = []
    for token, count in sorted(token_counts.items(), key=lambda x: (-x[1], x[0])):
        if token in brand_vocab or token in modifier_vocab or token in region_vocab:
            continue
        if count < min_token_count:
            continue
        examples = sorted(token_examples[token])
        novel_tokens.append(
            {
                "token": token,
                "count": int(count),
                "example_keywords": examples,
            }
        )

    novel_phrases: List[Dict] = []
    for phrase, count in sorted(phrase_counts.items(), key=lambda x: (-x[1], x[0])):
        if phrase in known_phrases:
            continue
        tokens = phrase.split()
        if all(tok in brand_vocab or tok in modifier_vocab or tok in region_vocab for tok in tokens):
            continue
        if count < min_phrase_count:
            continue
        examples = sorted(phrase_examples[phrase])
        novel_phrases.append(
            {
                "phrase": phrase,
                "count": int(count),
                "example_keywords": examples,
            }
        )

    return novel_tokens, novel_phrases


def evaluate_dataset(
    csv_in: str,
    truth_column: str = "expected_intent",
    output_dir: str = "reports",
    min_sim: float = 0.8,
    config_path: str = None,
    lexicon_path: str = None,
    min_token_count: int = 1,
    min_phrase_count: int = 2,
) -> Dict:
    """Run the pipeline on labelled data and capture evaluation artefacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_in)
    if truth_column not in df.columns:
        raise ValueError(f"Expected column '{truth_column}' in evaluation dataset.")

    brands, modifiers, regions = prepare_lexicon(config_path=config_path, lexicon_path=lexicon_path)
    processed = process_dataframe(df, min_sim=min_sim, lexicon=(brands, modifiers, regions))
    processed = processed.rename(
        columns={
            "intent": "predicted_intent",
            "intent_conf": "predicted_intent_conf",
        }
    )
    processed[truth_column] = df[truth_column]

    comparison = processed.copy()
    misclassified = comparison[comparison["predicted_intent"] != comparison[truth_column]]

    misclassified_path = output_path / "misclassified_examples.csv"
    columns = [
        col
        for col in [
            "keyword",
            "keyword_norm",
            truth_column,
            "predicted_intent",
            "predicted_intent_conf",
            "brands",
            "regions",
            "modifiers",
            "cluster_id",
            "centroid",
            "avg_sim",
        ]
        if col in misclassified.columns
    ]
    misclassified[columns].to_csv(misclassified_path, index=False)

    novel_tokens, novel_phrases = _collect_novel_candidates(
        processed,
        brands=brands,
        modifiers=modifiers,
        regions=regions,
        min_token_count=min_token_count,
        min_phrase_count=min_phrase_count,
    )

    novel_output = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source_csv": str(csv_in),
            "truth_column": truth_column,
            "total_rows": int(len(processed)),
            "misclassified_rows": int(len(misclassified)),
            "accuracy": (
                float((len(processed) - len(misclassified)) / len(processed))
                if len(processed)
                else None
            ),
        },
        "BRANDS": [],
        "MODIFIERS": [],
        "REGIONS": [],
        "UNMAPPED_TOKENS": novel_tokens,
        "UNMAPPED_PHRASES": novel_phrases,
    }

    novel_path = output_path / "novel_tokens.json"
    with open(novel_path, "w") as f:
        json.dump(novel_output, f, indent=2, ensure_ascii=False)

    accuracy = novel_output["metadata"]["accuracy"]

    return {
        "total_rows": len(processed),
        "misclassified_rows": len(misclassified),
        "accuracy": accuracy,
        "misclassified_path": str(misclassified_path),
        "novel_tokens_path": str(novel_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate clustering/intent predictions")
    parser.add_argument("csv_in", help="CSV file with labelled keywords")
    parser.add_argument("--truth-column", default="expected_intent")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--min-sim", type=float, default=0.8)
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--lexicon", dest="lexicon_path")
    parser.add_argument("--min-token-count", type=int, default=1)
    parser.add_argument("--min-phrase-count", type=int, default=2)
    args = parser.parse_args()

    summary = evaluate_dataset(
        csv_in=args.csv_in,
        truth_column=args.truth_column,
        output_dir=args.output_dir,
        min_sim=args.min_sim,
        config_path=args.config_path,
        lexicon_path=args.lexicon_path,
        min_token_count=args.min_token_count,
        min_phrase_count=args.min_phrase_count,
    )

    accuracy = summary["accuracy"]
    accuracy_msg = "N/A" if accuracy is None else f"{accuracy:.3f}"
    print(f"Processed {summary['total_rows']} rows | accuracy={accuracy_msg}")
    print(f"Misclassified examples -> {summary['misclassified_path']}")
    print(f"Novel token candidates -> {summary['novel_tokens_path']}")


if __name__ == "__main__":
    main()
