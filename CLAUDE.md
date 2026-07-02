# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A notebook-centered classical NLP analysis (not a production system) that classifies SMS spam using the UCI SMS Spam Collection. The importable package is `spam_classifier` (src layout); the distribution name in pyproject.toml is `sms-spam-classifier`.

## Commands

Use the repository-local virtual environment at .venv. If it does not exist, create it first. Activate it before running project commands.
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

```bash
source .venv/bin/activate
python -m pip install -e .          # setup (editable install; required for imports and tests)
python -m pytest -q                 # run all tests
python -m pytest tests/test_data.py::test_exact_duplicate_rows_are_removed  # run a single test
```

Re-execute the analysis notebook from the repo root (do not re-execute or modify the notebook for documentation, CI, test-only, or other changes that do not affect the analysis outputs):

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_sms_spam_classification.ipynb --ExecutePreprocessor.timeout=600
```

If `data/raw/SMSSpamCollection` is missing:

```bash
python -c "from spam_classifier.data import download_sms_spam_collection; download_sms_spam_collection()"
```

## Architecture

The notebook `notebooks/01_sms_spam_classification.ipynb` is the analysis entry point; all reusable logic lives in `src/spam_classifier/` and is imported by both the notebook and the tests:

- `data.py` — loading, validation, and deduplication. Defines `LABELS = ("ham", "spam")` and `POS_LABEL = "spam"`, which the other modules import. `validate_sms_dataframe` raises on unexpected labels or missing/empty messages, removes exact duplicate label/message rows, and returns a `DataValidationReport`. The download helper tries `ucimlrepo` (dataset ID 228) and falls back to UCI's static archive. The raw TSV is read with `keep_default_na=False` so message text is preserved exactly; the modeling workflow never alters the raw file.
- `modeling.py` — candidate models, hyperparameter grids, and model search. `RANDOM_STATE = 42` lives here and controls the single stratified 80/20 split and the 5-fold `StratifiedKFold`.
- `evaluation.py` — metrics that always treat spam as the positive class. `prediction_scores` normalizes `predict_proba` vs `decision_function` output so scores are spam-oriented regardless of class ordering.

## Invariants to preserve

- **Leakage safety**: `TfidfVectorizer` must stay inside each scikit-learn `Pipeline` so vocabulary/IDF are learned within each CV fold. Model selection uses only the training split; the holdout test set is evaluated exactly once. Label-informed EDA in the notebook uses training data only.
- **Spam is the positive class** for all metrics; model selection ranks by mean CV spam F1 (`refit="spam_f1"`), not accuracy.
- **Duplicates are removed before splitting**, so the split cannot place a duplicate pair across train and test.
- The README's model-comparison and holdout-results tables, row counts, and `images/*.png` reflect the executed notebook. If you change the data handling, models, or grids, re-execute the notebook and update the README numbers and images to match.
