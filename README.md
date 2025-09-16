# SEO Cluster & Intent Tool

This project groups keyword lists into semantic clusters, assigns intent labels, and now
includes an active-learning workflow for collecting human feedback on low-confidence
predictions.

## Getting started

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a CSV containing a `keyword` column. The app runs the clustering pipeline, shows a
preview of the results, and automatically queues any rows with an intent confidence below
your selected threshold.

## Active-learning label queue

- Low-confidence rows are written to `data/label_queue.csv` via
  `cluster/active_learning.py`.
- The Streamlit interface exposes this queue so you can edit the `human_intent` column
  inline and download the newly labeled rows as a CSV.
- Use the **Save queue updates** button to persist changes back to disk.

The queue path is tracked in `.gitignore` because it will evolve as your team adds new
labels.

## Weekly retraining workflow

New labels can be folded into a lightweight text classifier to benchmark the heuristic
intent rules.

```bash
python scripts/weekly_retrain.py  # runs a single retraining/evaluation cycle
```

The script will:

1. Load the existing training set from `data/training_data.csv`.
2. Pull human labels from `data/label_queue.csv`.
3. Hold out part of the new labels for evaluation, retrain a TF-IDF + logistic regression
   model, and print before/after macro F1 scores.
4. Append all labeled examples back into `data/training_data.csv` for future runs.

To keep the model refreshed automatically, run the script with `--schedule` and place it
under a process manager (or container) that stays online:

```bash
python scripts/weekly_retrain.py --schedule --run-day sunday --run-time 02:00
```

The scheduler computes the next run time, sleeps until then, and repeats weekly. Use
`Ctrl+C` to stop the loop.

## Repository structure

```
app.py                      # Streamlit interface
cluster/pipeline.py         # keyword clustering & heuristic intent tagging
cluster/active_learning.py  # low-confidence queue helpers
scripts/weekly_retrain.py   # weekly retraining & scheduler entry point
data/training_data.csv      # bootstrap training examples
```
