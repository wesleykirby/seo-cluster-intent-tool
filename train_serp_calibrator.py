"""Command line utility for training the SERP logistic calibrator."""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from cluster.pipeline import BRANDS
from cluster.serp_calibrator import DEFAULT_MODEL_FILENAME, SERPCalibrator

LOGGER = logging.getLogger(__name__)


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_brand_aliases(config_path: Optional[str]) -> List[str]:
    brand_map = dict(BRANDS)
    if config_path:
        with open(config_path) as fp:
            cfg = json.load(fp)
        brand_map.update(cfg.get("BRANDS", {}))
    aliases = set(brand_map.keys()) | set(brand_map.values())
    return sorted({alias.strip().lower() for alias in aliases if alias})


def _log_split_metrics(split: str, y_true: np.ndarray, probs: np.ndarray) -> None:
    preds = (probs >= 0.5).astype(int)
    accuracy = accuracy_score(y_true, preds)
    clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    try:
        roc_auc = roc_auc_score(y_true, probs)
    except ValueError:
        roc_auc = math.nan
    loss = log_loss(y_true, clipped)
    brier = brier_score_loss(y_true, probs)
    LOGGER.info(
        "%s calibration metrics: accuracy=%.3f roc_auc=%s log_loss=%.3f brier=%.3f",
        split,
        accuracy,
        "nan" if math.isnan(roc_auc) else f"{roc_auc:.3f}",
        loss,
        brier,
    )


def train(args: argparse.Namespace) -> None:
    _configure_logging(args.log_level)
    data_path = Path(args.train_csv)
    if not data_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' missing from training data")

    brands = _load_brand_aliases(args.config)
    calibrator = SERPCalibrator(brands=brands)

    LOGGER.info("Loaded %d training rows", len(df))
    if args.test_size and 0 < args.test_size < 1:
        stratify = df[args.label_col] if df[args.label_col].nunique() > 1 else None
        train_df, valid_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify,
        )
    else:
        train_df, valid_df = df, None

    calibrator.fit(train_df, label_col=args.label_col)
    train_probs = calibrator.predict_proba(train_df)
    _log_split_metrics("Train", train_df[args.label_col].to_numpy(), train_probs.to_numpy())

    if valid_df is not None and not valid_df.empty:
        valid_probs = calibrator.predict_proba(valid_df)
        _log_split_metrics("Validation", valid_df[args.label_col].to_numpy(), valid_probs.to_numpy())

    output_path = Path(args.model_out or Path("cluster") / DEFAULT_MODEL_FILENAME)
    calibrator.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SERP logistic calibrator")
    parser.add_argument("train_csv", help="CSV file with SERP metadata and binary labels")
    parser.add_argument(
        "--label-col",
        default="label",
        help="Column name containing the binary target (1=desired intent)",
    )
    parser.add_argument(
        "--model-out",
        default=Path("cluster") / DEFAULT_MODEL_FILENAME,
        help="Path where the trained model should be stored",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split fraction (set to 0 to disable validation)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--config",
        help="Optional JSON config with BRANDS overrides used for feature extraction",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
