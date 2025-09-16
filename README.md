# seo-cluster-intent-tool

Utilities for clustering SEO keywords, assigning intents, and maintaining the lexical resources that drive the tagging heuristics.

## Keyword pipeline

Run the core pipeline from the command line:

```bash
python -m cluster.pipeline keywords_in.csv keywords_tagged.csv --min-sim 0.8
```

The pipeline expects a CSV with a `keyword` column and writes the augmented results to the output path.

## Evaluation & maintenance workflow

Maintaining high intent accuracy requires a feedback loop. The project now ships with tooling to capture misclassifications, surface unseen vocabulary, and append vetted tokens to the shared lexicon.

### 1. Evaluate predictions & capture artefacts

Provide a labelled CSV with a `keyword` column and a ground-truth intent column (default name: `expected_intent`). Then run:

```bash
python -m cluster.evaluation labelled_keywords.csv --truth-column expected_intent --output-dir reports
```

The command writes two artefacts under the chosen output directory:

* `misclassified_examples.csv` – rows where the predicted intent differs from the supplied truth. Columns include the original keyword, normalised form, predicted intent/confidence, extracted tags, and cluster context.
* `novel_tokens.json` – a structured report containing:
  * Empty `BRANDS`, `MODIFIERS`, and `REGIONS` arrays ready for reviewed additions.
  * `UNMAPPED_TOKENS` and `UNMAPPED_PHRASES` lists summarising unknown vocabulary with counts and sample keywords to help review.

Use `--min-sim`, `--min-token-count`, and `--min-phrase-count` to tune the evaluation if needed. Runtime brand/modifier/region overrides can still be provided through `--config` or by pointing to an alternate lexicon with `--lexicon`.

### 2. Review `novel_tokens.json`

Inspect the `UNMAPPED_*` sections and decide which entries should extend the lexicon. Promote approved items into the appropriate top-level arrays:

```json
{
  "BRANDS": [
    {"alias": "betway gh", "canonical": "betway"},
    "mybet"
  ],
  "MODIFIERS": ["sports picks"],
  "REGIONS": ["nigeria"],
  "UNMAPPED_TOKENS": [...],
  "UNMAPPED_PHRASES": [...]
}
```

* Brand entries can be either strings (alias equals canonical form) or objects with explicit `alias`/`canonical` keys for synonym mapping.
* Modifiers and regions are simple strings. All values are normalised to lower-case when applied.

Leave unapproved candidates in the `UNMAPPED_*` sections for later review.

### 3. Append reviewed tokens to the shared lexicon

Run the updater to merge the reviewed entries into `cluster/lexicon.json` while avoiding duplicates:

```bash
python update_lexicon.py --tokens reports/novel_tokens.json
```

Add `--dry-run` to preview the changes without writing to disk. The script reports which brands, modifiers, and regions were appended. Subsequent runs of the pipeline automatically pick up the refreshed lexicon.

After applying updates you can clear the promoted entries from `novel_tokens.json` (optional) to keep the review queue tidy.

## Lexicon structure

The canonical lexicon lives at `cluster/lexicon.json`. It stores:

* `BRANDS`: alias → canonical mappings used during keyword normalisation.
* `MODIFIERS`: phrases matched as modifiers.
* `REGIONS`: supported geographies.

The pipeline loads this file on each execution, so committed changes take effect immediately. You can still provide ad-hoc overrides via the existing `--config` argument when running the pipeline.
