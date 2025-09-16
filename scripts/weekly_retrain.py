"""Automate weekly retraining using newly labelled keywords."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster.active_learning import DEFAULT_QUEUE_PATH, load_label_queue
from cluster.pipeline import normalize_kw

DEFAULT_TRAINING_PATH = Path("data/training_data.csv")


def _ensure_keyword_norm(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if "keyword" not in df.columns:
        raise ValueError("Expected a 'keyword' column in the training data")
    result = df.copy()
    if "keyword_norm" not in result.columns:
        result["keyword_norm"] = result["keyword"].astype(str).apply(normalize_kw)
    else:
        result["keyword_norm"] = result["keyword_norm"].astype(str)
    return result


def load_training_data(path: Path = DEFAULT_TRAINING_PATH) -> pd.DataFrame:
    if not path.exists():
        print(f"[weekly_retrain] No training data found at {path}. Starting fresh.")
        return pd.DataFrame(columns=["keyword", "intent"])
    df = pd.read_csv(path)
    missing = {"keyword", "intent"} - set(df.columns)
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")
    return df


def load_new_labels(queue_path: Path = DEFAULT_QUEUE_PATH) -> pd.DataFrame:
    queue_df = load_label_queue(queue_path)
    if queue_df.empty or "human_intent" not in queue_df.columns:
        return pd.DataFrame(columns=["keyword", "intent"])
    mask = queue_df["human_intent"].astype(str).str.strip() != ""
    labeled = queue_df.loc[mask].copy()
    if labeled.empty:
        return pd.DataFrame(columns=["keyword", "intent"])
    labeled.rename(columns={"human_intent": "intent"}, inplace=True)
    keep_cols = [c for c in ["keyword", "intent", "keyword_norm"] if c in labeled.columns]
    return labeled[keep_cols]


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            (
                "clf",
                LogisticRegression(max_iter=1000, multi_class="auto"),
            ),
        ]
    )


def _feature_column(df: pd.DataFrame) -> pd.Series:
    if "keyword_norm" in df.columns:
        return df["keyword_norm"].astype(str)
    return df["keyword"].astype(str)


def evaluate_model(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> Optional[float]:
    if train_df.empty or eval_df.empty:
        return None
    if train_df["intent"].nunique() < 2:
        return None
    model = build_model()
    model.fit(_feature_column(train_df), train_df["intent"])
    predictions = model.predict(_feature_column(eval_df))
    labels = sorted(eval_df["intent"].unique())
    return float(f1_score(eval_df["intent"], predictions, labels=labels, average="macro"))


def _split_new_labels(df: pd.DataFrame, holdout_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or len(df) == 1:
        return pd.DataFrame(columns=df.columns), df.copy()
    test_size = max(1, int(round(len(df) * holdout_ratio)))
    if test_size >= len(df):
        test_size = len(df) - 1
    train_df, eval_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def run_retraining(
    training_path: Path = DEFAULT_TRAINING_PATH,
    queue_path: Path = DEFAULT_QUEUE_PATH,
    holdout_ratio: float = 0.4,
) -> Dict[str, Union[None, float, int]]:
    base_data = _ensure_keyword_norm(load_training_data(training_path))
    new_labels = _ensure_keyword_norm(load_new_labels(queue_path))

    if new_labels.empty:
        print("[weekly_retrain] No newly labeled keywords found. Skipping retraining.")
        return {"before_f1": None, "after_f1": None, "evaluated_rows": 0, "added_rows": 0}

    train_new, eval_new = _split_new_labels(new_labels, holdout_ratio)
    evaluation_set = eval_new if not eval_new.empty else new_labels

    before_score = evaluate_model(base_data, evaluation_set)

    augmented_training = pd.concat([base_data, train_new], ignore_index=True)
    if augmented_training.empty:
        augmented_training = evaluation_set.copy()
    augmented_training = _ensure_keyword_norm(augmented_training)
    after_score = evaluate_model(augmented_training, evaluation_set)

    updated_training = pd.concat([base_data, new_labels], ignore_index=True)
    if not updated_training.empty:
        updated_training = _ensure_keyword_norm(updated_training)
        updated_training = updated_training.drop_duplicates(subset=["keyword_norm", "intent"], keep="last")
        updated_training.to_csv(training_path, index=False)

    if before_score is None:
        print("[weekly_retrain] Before retraining F1: n/a")
    else:
        print(f"[weekly_retrain] Before retraining F1: {before_score:.3f}")
    if after_score is None:
        print("[weekly_retrain] After retraining F1: n/a")
    else:
        print(f"[weekly_retrain] After retraining F1: {after_score:.3f}")
    print(f"[weekly_retrain] Evaluated on {len(evaluation_set)} labeled keywords.")

    return {
        "before_f1": before_score,
        "after_f1": after_score,
        "evaluated_rows": len(evaluation_set),
        "added_rows": len(new_labels),
    }


def _compute_next_run(run_day: str, run_time: str) -> datetime:
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if run_day.lower() not in days:
        raise ValueError(f"Unknown run day '{run_day}'. Expected one of {days}.")
    hour, minute = (int(part) for part in run_time.split(":"))
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    day_offset = (days.index(run_day.lower()) - now.weekday()) % 7
    next_run = target + timedelta(days=day_offset)
    if next_run <= now:
        next_run += timedelta(days=7)
    return next_run


def schedule_weekly(
    training_path: Path,
    queue_path: Path,
    holdout_ratio: float,
    run_day: str,
    run_time: str,
) -> None:
    print(
        f"[weekly_retrain] Scheduling weekly retraining for every {run_day} at {run_time}."
    )
    try:
        while True:
            next_run = _compute_next_run(run_day, run_time)
            sleep_seconds = (next_run - datetime.now()).total_seconds()
            print(f"[weekly_retrain] Next run scheduled for {next_run.isoformat()}.")
            time.sleep(max(0, sleep_seconds))
            run_retraining(training_path=training_path, queue_path=queue_path, holdout_ratio=holdout_ratio)
    except KeyboardInterrupt:
        print("[weekly_retrain] Scheduler stopped by user.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly retraining utility")
    parser.add_argument("--training-path", type=Path, default=DEFAULT_TRAINING_PATH)
    parser.add_argument("--queue-path", type=Path, default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--holdout-ratio", type=float, default=0.4)
    parser.add_argument("--run-day", type=str, default="sunday", help="Day of week to trigger retraining (e.g. sunday)")
    parser.add_argument("--run-time", type=str, default="02:00", help="24h time to trigger retraining (HH:MM)")
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="If provided, keep running and trigger retraining weekly instead of once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.schedule:
        schedule_weekly(
            training_path=args.training_path,
            queue_path=args.queue_path,
            holdout_ratio=args.holdout_ratio,
            run_day=args.run_day,
            run_time=args.run_time,
        )
    else:
        run_retraining(
            training_path=args.training_path,
            queue_path=args.queue_path,
            holdout_ratio=args.holdout_ratio,
        )


if __name__ == "__main__":
    main()
