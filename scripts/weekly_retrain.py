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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster.active_learning import DEFAULT_QUEUE_PATH, load_label_queue
from cluster.pipeline import normalize_kw
from cluster.vector_semantic_learner import VectorSemanticLearner

# Always resolve the canonical training data relative to the project root so
# uploads performed from other working directories (for example when the
# Streamlit app is launched via an absolute path) still persist to the
# repository copy of the CSV.
DEFAULT_TRAINING_PATH = ROOT / "data" / "training_data.csv"


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
        # Return both old and new format columns for compatibility
        return pd.DataFrame(columns=["keyword", "intent", "main_topic", "sub_topic", "modifier"])
    df = pd.read_csv(path)
    
    # Support both old format (keyword,intent) and new format (keyword,main_topic,sub_topic,modifier)
    if "main_topic" in df.columns and "sub_topic" in df.columns and "modifier" in df.columns:
        # New enhanced format - validate required columns
        required_cols = {"keyword", "main_topic", "sub_topic", "modifier"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Enhanced training data is missing required columns: {missing}")
        # Add intent column for backward compatibility if not present
        if "intent" not in df.columns:
            df["intent"] = "semantic"  # Placeholder for legacy compatibility
    else:
        # Legacy format - validate old required columns
        missing = {"keyword", "intent"} - set(df.columns)
        if missing:
            raise ValueError(f"Training data is missing required columns: {missing}")
        # Add new columns with default values for compatibility
        if "main_topic" not in df.columns:
            df["main_topic"] = "Betting"
        if "sub_topic" not in df.columns:
            df["sub_topic"] = "General"
        if "modifier" not in df.columns:
            df["modifier"] = "General"
    
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
    """Legacy model builder for backward compatibility with old intent classification."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            (
                "clf",
                LogisticRegression(max_iter=1000, multi_class="auto"),
            ),
        ]
    )


def train_vector_semantic_model(training_df: pd.DataFrame) -> Dict[str, Union[None, float, int]]:
    """
    Train the enhanced vector-based semantic model.
    This function trains on the full semantic structure (Main/Sub/Mod) instead of just basic intent.
    """
    # Initialize the vector semantic learner
    learner = VectorSemanticLearner()
    
    # Check if we have enhanced training data
    if "main_topic" in training_df.columns and "sub_topic" in training_df.columns and "modifier" in training_df.columns:
        print("[weekly_retrain] Training enhanced vector-based semantic model...")
        
        # Train on the full semantic structure
        results = learner.train(training_df)
        
        if results['status'] == 'success':
            print(f"[weekly_retrain] Vector semantic model trained successfully!")
            print(f"  - Main Topic Accuracy: {results['main_topic_accuracy']:.3f}")
            print(f"  - Sub Topic Accuracy: {results['sub_topic_accuracy']:.3f}")
            print(f"  - Modifier Accuracy: {results['modifier_accuracy']:.3f}")
            print(f"  - Learned {results['learned_brands']} brands from training data")
            print(f"  - Discovered {results['learned_patterns']} semantic patterns")
            
            return {
                "vector_training_status": "success",
                "main_accuracy": results['main_topic_accuracy'],
                "sub_accuracy": results['sub_topic_accuracy'],
                "modifier_accuracy": results['modifier_accuracy'],
                "learned_brands": results['learned_brands'],
                "learned_patterns": results['learned_patterns'],
                "training_samples": results['training_samples']
            }
        else:
            print(f"[weekly_retrain] Vector semantic training failed: {results.get('reason', 'Unknown error')}")
            return {"vector_training_status": "failed", "reason": results.get('reason', 'Unknown error')}
    else:
        print("[weekly_retrain] No enhanced training data found (missing main_topic/sub_topic/modifier columns)")
        print("[weekly_retrain] Skipping vector semantic model training")
        return {"vector_training_status": "skipped", "reason": "No enhanced training data"}


def evaluate_enhanced_model(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Evaluate the enhanced vector-based semantic model.
    Returns accuracy scores for each semantic component.
    """
    if train_df.empty or eval_df.empty:
        return None
    
    # Check if we have enhanced training data
    required_cols = ["main_topic", "sub_topic", "modifier"]
    if not all(col in train_df.columns and col in eval_df.columns for col in required_cols):
        return None
    
    # Train temporary model for evaluation
    learner = VectorSemanticLearner()
    train_results = learner.train(train_df)
    
    if train_results['status'] != 'success':
        return None
    
    # Predict on evaluation set
    keywords = eval_df['keyword'].astype(str).tolist()
    predictions = learner.predict(keywords)
    
    # Calculate accuracies
    y_true_main = eval_df['main_topic'].tolist()
    y_true_sub = eval_df['sub_topic'].tolist()
    y_true_mod = eval_df['modifier'].tolist()
    
    y_pred_main = [pred['main_topic'] for pred in predictions]
    y_pred_sub = [pred['sub_topic'] for pred in predictions]
    y_pred_mod = [pred['modifier'] for pred in predictions]
    
    main_acc = accuracy_score(y_true_main, y_pred_main)
    sub_acc = accuracy_score(y_true_sub, y_pred_sub)
    mod_acc = accuracy_score(y_true_mod, y_pred_mod)
    
    return {
        "main_accuracy": main_acc,
        "sub_accuracy": sub_acc,
        "modifier_accuracy": mod_acc,
        "average_accuracy": (main_acc + sub_acc + mod_acc) / 3
    }


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

    # Check if we have any training data (new labels OR existing enhanced data)
    has_enhanced_data = "main_topic" in base_data.columns and not base_data.empty
    has_new_labels = not new_labels.empty
    
    if not has_new_labels and not has_enhanced_data:
        print("[weekly_retrain] No training data found. Skipping retraining.")
        return {"before_f1": None, "after_f1": None, "evaluated_rows": 0, "added_rows": 0, "vector_training_status": "skipped"}

    # Initialize results dictionary
    results = {"vector_training_status": "skipped", "vector_stats": {}}

    # If we have enhanced training data, train the vector-based semantic model
    if has_enhanced_data:
        print("[weekly_retrain] Enhanced training data detected - training vector-based semantic model")
        vector_results = train_vector_semantic_model(base_data)
        results.update(vector_results)
        results["vector_stats"] = vector_results

    # Legacy intent classification training (for backward compatibility)
    if has_new_labels:
        print("[weekly_retrain] Processing new labels with legacy intent classification")
        train_new, eval_new = _split_new_labels(new_labels, holdout_ratio)
        evaluation_set = eval_new if not eval_new.empty else new_labels

        before_score = evaluate_model(base_data, evaluation_set)

        augmented_training = pd.concat([base_data, train_new], ignore_index=True)
        if augmented_training.empty:
            augmented_training = evaluation_set.copy()
        augmented_training = _ensure_keyword_norm(augmented_training)
        after_score = evaluate_model(augmented_training, evaluation_set)

        # Save updated training data
        updated_training = pd.concat([base_data, new_labels], ignore_index=True)
        if not updated_training.empty:
            updated_training = _ensure_keyword_norm(updated_training)
            # Drop duplicates based on available columns
            if "main_topic" in updated_training.columns:
                # Enhanced format - use semantic deduplication
                updated_training = updated_training.drop_duplicates(subset=["keyword_norm"], keep="last")
            else:
                # Legacy format - use intent deduplication
                updated_training = updated_training.drop_duplicates(subset=["keyword_norm", "intent"], keep="last")
            updated_training.to_csv(training_path, index=False)

        # Update results with legacy training metrics
        results.update({
            "before_f1": before_score,
            "after_f1": after_score,
            "evaluated_rows": len(evaluation_set),
            "added_rows": len(new_labels),
        })

        # Enhanced evaluation if we have the right data
        enhanced_eval = evaluate_enhanced_model(augmented_training, evaluation_set)
        if enhanced_eval:
            results["enhanced_evaluation"] = enhanced_eval
            print(f"[weekly_retrain] Enhanced model evaluation:")
            print(f"  - Main Topic Accuracy: {enhanced_eval['main_accuracy']:.3f}")
            print(f"  - Sub Topic Accuracy: {enhanced_eval['sub_accuracy']:.3f}")
            print(f"  - Modifier Accuracy: {enhanced_eval['modifier_accuracy']:.3f}")

        if before_score is None:
            print("[weekly_retrain] Before retraining F1: n/a")
        else:
            print(f"[weekly_retrain] Before retraining F1: {before_score:.3f}")
        if after_score is None:
            print("[weekly_retrain] After retraining F1: n/a")
        else:
            print(f"[weekly_retrain] After retraining F1: {after_score:.3f}")
        print(f"[weekly_retrain] Evaluated on {len(evaluation_set)} labeled keywords.")
    else:
        # No new labels, but we have enhanced data - just train the vector model
        results.update({
            "before_f1": None,
            "after_f1": None,
            "evaluated_rows": 0,
            "added_rows": 0,
        })
        print("[weekly_retrain] No new labels to process, but vector-based semantic model was updated")

    return results


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
