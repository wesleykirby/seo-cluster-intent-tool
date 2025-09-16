from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from cluster.embedding import DEFAULT_BASE_MODEL


TRIPLET_KEYS = {"anchor", "positive", "negative"}


def _validate_record(record: dict[str, str], idx: int) -> tuple[str, str, str]:
    missing = TRIPLET_KEYS.difference(record)
    if missing:
        raise ValueError(f"Missing keys {missing} in record {idx}")
    return record["anchor"], record["positive"], record["negative"]


def load_triplets(path: Path) -> Iterator[InputExample]:
    """Yield :class:`InputExample` instances from a dataset file.

    The loader accepts CSV/TSV files with ``anchor``, ``positive`` and
    ``negative`` columns as well as JSON or JSONL files following the same key
    structure.
    """

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path)
        if not TRIPLET_KEYS.issubset(df.columns):
            missing = TRIPLET_KEYS.difference(df.columns)
            raise ValueError(f"Dataset is missing required columns: {missing}")
        for record in df[["anchor", "positive", "negative"]].itertuples(index=False):
            yield InputExample(texts=list(record))
        return

    if suffix == ".jsonl":
        with path.open() as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                payload = json.loads(line)
                triplet = _validate_record(payload, idx)
                yield InputExample(texts=list(triplet))
        return

    if suffix == ".json":
        payload = json.loads(path.read_text())
        if not isinstance(payload, (list, tuple)):
            raise ValueError("JSON dataset must contain an iterable of records")
        for idx, record in enumerate(payload):
            triplet = _validate_record(record, idx)
            yield InputExample(texts=list(triplet))
        return

    raise ValueError(
        "Unsupported dataset format. Provide CSV/TSV, JSON, or JSONL files with"
        " 'anchor', 'positive', and 'negative' fields."
    )


def build_dataloader(examples: list[InputExample], batch_size: int) -> DataLoader:
    if not examples:
        raise ValueError("The training dataset is empty. Provide at least one triplet.")
    return DataLoader(examples, shuffle=True, batch_size=batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the clustering encoder with triplet loss")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to the curated triplet dataset")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where the fine-tuned model will be stored")
    parser.add_argument(
        "--pretrained-model",
        default=DEFAULT_BASE_MODEL,
        help="Base sentence-transformer model to start from",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of total steps used for warmup",
    )
    args = parser.parse_args()

    examples = list(load_triplets(args.dataset))
    dataloader = build_dataloader(examples, args.batch_size)

    model = SentenceTransformer(args.pretrained_model)
    train_loss = losses.TripletLoss(
        model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=0.3,
    )

    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=args.epochs,
        optimizer_params={"lr": args.learning_rate},
        warmup_steps=warmup_steps,
        scheduler="warmuplinear",
        show_progress_bar=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output_dir))


if __name__ == "__main__":
    main()
