#!/usr/bin/env python
"""Train the vertical router using weak supervision and self-training."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from cluster.pipeline import normalize_kw
from cluster.router import (
    DEFAULT_VERTICAL_RULES,
    VerticalRouter,
    build_rule_feature_matrix,
    weak_rule_labels,
)


def _encode_texts(
    texts: List[str],
    model_name: str,
    *,
    batch_size: int | None = None,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    if normalize_embeddings:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        embeddings = embeddings / norms
    return embeddings.astype(np.float32)


def _make_feature_matrix(
    texts: List[str],
    embeddings: np.ndarray,
) -> np.ndarray:
    rule_feats = build_rule_feature_matrix(texts, DEFAULT_VERTICAL_RULES)
    if rule_feats.size == 0:
        return embeddings
    return np.hstack([embeddings, rule_feats]).astype(np.float32)


def train(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)
    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found in {args.input}")

    raw_texts = df[args.text_column].fillna("").astype(str)
    normalized = raw_texts.apply(normalize_kw)
    texts = normalized.tolist()

    print(f"Encoding {len(texts)} keywords with {args.model_name} ...")
    embeddings = _encode_texts(
        texts,
        args.model_name,
        batch_size=args.batch_size,
        normalize_embeddings=args.normalize_embeddings,
    )
    features = _make_feature_matrix(texts, embeddings)

    weak_labels, weak_conf, weak_strength = weak_rule_labels(texts, DEFAULT_VERTICAL_RULES)
    weak_labels_arr = np.array(weak_labels, dtype=object)
    weak_conf_arr = np.array(weak_conf, dtype=float)
    weak_strength_arr = np.array(weak_strength, dtype=float)

    mask = (
        weak_labels_arr != None  # noqa: E711 - intentional comparison
        ) & (weak_conf_arr >= args.weak_threshold) & (weak_strength_arr >= args.min_rule_strength)

    labeled_count = int(mask.sum())
    print(f"Found {labeled_count} high-confidence weak labels out of {len(texts)} keywords.")
    if labeled_count == 0:
        raise RuntimeError("No training data met the weak labeling thresholds.")

    label_encoder = LabelEncoder()
    y_initial = label_encoder.fit_transform(weak_labels_arr[mask])
    X_labeled = features[mask]

    unique_labels = np.unique(y_initial)
    if unique_labels.size < 2:
        print("Only one weak label class found; using DummyClassifier.")
        model = DummyClassifier(strategy="most_frequent")
    else:
        model = LogisticRegression(
            max_iter=args.max_iter,
            class_weight="balanced",
            random_state=args.random_state,
            multi_class="auto",
        )
    model.fit(X_labeled, y_initial)

    labeled_mask = mask.copy()
    y_labeled = y_initial.copy()
    X_current = X_labeled.copy()

    if isinstance(model, LogisticRegression) and args.self_train_steps > 0:
        for step in range(1, args.self_train_steps + 1):
            probs = model.predict_proba(features)
            max_prob = probs.max(axis=1)
            best_idx = probs.argmax(axis=1)
            pseudo_mask = (max_prob >= args.self_train_threshold) & (~labeled_mask)
            added = int(pseudo_mask.sum())
            if added == 0:
                print(f"Self-training step {step}: no pseudo labels above threshold.")
                break
            print(f"Self-training step {step}: adding {added} pseudo-labeled samples.")
            X_pseudo = features[pseudo_mask]
            y_pseudo = model.classes_[best_idx[pseudo_mask]]
            X_current = np.vstack([X_current, X_pseudo])
            y_labeled = np.concatenate([y_labeled, y_pseudo])
            labeled_mask[pseudo_mask] = True
            model.fit(X_current, y_labeled)

    router = VerticalRouter(
        model=model,
        label_encoder=label_encoder,
        rules=DEFAULT_VERTICAL_RULES,
        embedder_name=args.model_name,
        min_confidence=args.inference_threshold,
        fallback_label=args.fallback_label,
        normalize_embeddings=args.normalize_embeddings,
        embedding_batch_size=args.batch_size,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    router.save(output_path)
    print(f"Router saved to {output_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the keyword vertical router")
    parser.add_argument("--input", default="keywords_in.csv", help="Training CSV file")
    parser.add_argument(
        "--text-column",
        default="keyword",
        help="Name of the column containing keyword text",
    )
    parser.add_argument("--output", default="router.joblib", help="Where to save the trained router")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument(
        "--weak-threshold",
        type=float,
        default=0.6,
        help="Minimum weak-label confidence ratio to include in supervised set",
    )
    parser.add_argument(
        "--min-rule-strength",
        type=float,
        default=1.0,
        help="Minimum aggregated rule score to treat as labeled",
    )
    parser.add_argument(
        "--self-train-threshold",
        type=float,
        default=0.9,
        help="Probability threshold for accepting pseudo labels",
    )
    parser.add_argument(
        "--self-train-steps",
        type=int,
        default=3,
        help="Maximum number of self-training iterations",
    )
    parser.add_argument("--max-iter", type=int, default=1000, help="Max iterations for LogisticRegression")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for the classifier")
    parser.add_argument(
        "--inference-threshold",
        type=float,
        default=0.5,
        help="Minimum probability to keep the predicted label at inference",
    )
    parser.add_argument(
        "--fallback-label",
        default="general",
        help="Label to assign when confidence is below threshold",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        dest="normalize_embeddings",
        action="store_false",
        help="Disable embedding L2 normalization",
    )
    parser.set_defaults(normalize_embeddings=True)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    train(parser.parse_args())
