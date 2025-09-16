"""Train vertical-specific intent classifiers with optional self-training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cluster.models.intent import IntentFeatureBuilder, IntentModel
from cluster.text_utils import INTENT_RULES, apply_domain_config, normalize_kw


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "keyword" not in df.columns:
        raise ValueError("Training data must include a 'keyword' column.")
    if "vertical" not in df.columns:
        raise ValueError("Training data must include a 'vertical' column.")
    if "intent" not in df.columns:
        df["intent"] = pd.NA
    df = df.copy()
    df["keyword"] = df["keyword"].astype(str)
    df["keyword_norm"] = df["keyword"].apply(normalize_kw)
    df["vertical"] = df["vertical"].astype(str)
    df["intent"] = df["intent"].astype("string")
    return df


def _split_labeled(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask = df["intent"].notna() & df["intent"].str.strip().ne("")
    labeled = df.loc[mask].copy()
    unlabeled = df.loc[~mask].copy()
    return labeled, unlabeled


def train_vertical_model(
    data: pd.DataFrame,
    iterations: int,
    threshold: float,
    min_labeled: int,
    max_iter: int,
) -> Tuple[Optional[IntentModel], Dict[str, object]]:
    labeled, unlabeled = _split_labeled(data)
    summary: Dict[str, object] = {
        "num_samples": int(len(data)),
        "num_labeled": int(len(labeled)),
        "num_unlabeled": int(len(unlabeled)),
    }
    if len(labeled) < max(min_labeled, 2):
        summary["skipped"] = "not_enough_labelled_examples"
        return None, summary
    if labeled["intent"].nunique() < 2:
        summary["skipped"] = "need_at_least_two_intents"
        return None, summary

    builder = IntentFeatureBuilder(intent_rules=INTENT_RULES)
    pseudo_total = 0
    iterations_run = 0

    for iteration in range(max(iterations, 1)):
        fit_frame = pd.concat([labeled, unlabeled], ignore_index=True) if not unlabeled.empty else labeled
        builder.fit(fit_frame)

        X_train = builder.transform(labeled)
        y_train = labeled["intent"].astype(str)
        classifier = LogisticRegression(max_iter=max_iter, multi_class="auto")
        classifier.fit(X_train, y_train)
        iterations_run = iteration + 1

        if unlabeled.empty:
            break
        X_unlabeled = builder.transform(unlabeled)
        proba = classifier.predict_proba(X_unlabeled)
        max_prob = proba.max(axis=1)
        best_idx = proba.argmax(axis=1)
        high_conf_mask = max_prob >= threshold
        if not np.any(high_conf_mask):
            break
        selected = unlabeled.iloc[np.where(high_conf_mask)[0]].copy()
        preds = classifier.classes_[best_idx[high_conf_mask]]
        selected.loc[:, "intent"] = preds
        selected.loc[:, "_pseudo_label"] = True

        labeled = pd.concat([labeled, selected], ignore_index=False)
        unlabeled = unlabeled.drop(selected.index)
        pseudo_total += len(selected)

    # Final fit to ensure classifier is trained on augmented set
    fit_frame = pd.concat([labeled, unlabeled], ignore_index=True) if not unlabeled.empty else labeled
    builder.fit(fit_frame)
    X_train = builder.transform(labeled)
    y_train = labeled["intent"].astype(str)
    classifier = LogisticRegression(max_iter=max_iter, multi_class="auto")
    classifier.fit(X_train, y_train)

    metadata = {
        "classes": classifier.classes_.tolist(),
        "iterations_trained": iterations_run,
        "pseudo_labeled": pseudo_total,
        "num_labeled_final": int(len(labeled)),
        "serp_columns": builder.serp_columns_,
    }
    summary.update(metadata)

    model = IntentModel(feature_builder=builder, classifier=classifier, metadata=metadata)
    return model, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train intent models per vertical")
    parser.add_argument("train_csv", help="CSV containing labelled training data")
    parser.add_argument("--unlabeled-csv", dest="unlabeled_csv", help="Optional CSV of additional unlabeled data")
    parser.add_argument("--output-dir", dest="output_dir", default="cluster/models/intent", help="Directory to save trained models")
    parser.add_argument("--config", dest="config_path", help="Optional config JSON to extend domain dictionaries")
    parser.add_argument("--self-train-iterations", type=int, default=3, help="Maximum self-training iterations")
    parser.add_argument("--self-train-threshold", type=float, default=0.9, help="Confidence required to include pseudo labels")
    parser.add_argument("--min-labeled", type=int, default=10, help="Minimum labelled rows required per vertical")
    parser.add_argument("--max-iter", type=int, default=500, help="Max iterations for the logistic regression classifier")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing model artifacts")
    args = parser.parse_args()

    if args.config_path:
        with open(args.config_path) as fh:
            config = json.load(fh)
        apply_domain_config(config)

    train_df = pd.read_csv(args.train_csv)
    frames = [train_df]
    if args.unlabeled_csv:
        unlabeled_df = pd.read_csv(args.unlabeled_csv)
        if "intent" not in unlabeled_df.columns:
            unlabeled_df["intent"] = pd.NA
        frames.append(unlabeled_df)
    data = pd.concat(frames, ignore_index=True, sort=False)
    data = _prepare_dataframe(data)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for vertical in sorted(data["vertical"].dropna().unique()):
        model_path = output_dir / f"{vertical}.joblib"
        if model_path.exists() and not args.overwrite:
            print(f"Skipping {vertical}: artifact already exists at {model_path}")
            continue

        vertical_data = data[data["vertical"] == vertical].copy()
        model, summary = train_vertical_model(
            vertical_data,
            iterations=args.self_train_iterations,
            threshold=args.self_train_threshold,
            min_labeled=args.min_labeled,
            max_iter=args.max_iter,
        )
        summaries[vertical] = summary
        if model is None:
            reason = summary.get("skipped", "training_skipped")
            print(f"Skipping {vertical}: {reason}")
            continue

        joblib.dump(model, model_path)
        classes = summary.get("classes", [])
        print(
            f"Saved model for vertical '{vertical}' with classes={classes} at {model_path}")

    if summaries:
        report_path = output_dir / "training_report.json"
        with open(report_path, "w") as fh:
            json.dump(summaries, fh, indent=2)
        print(f"Wrote training summary to {report_path}")


if __name__ == "__main__":
    main()
