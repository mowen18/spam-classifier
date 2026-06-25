"""Data loading and validation for the UCI SMS Spam Collection."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

LABELS = ("ham", "spam")
POS_LABEL = "spam"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "SMSSpamCollection"
UCI_DATASET_ID = 228
UCI_ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
)


@dataclass(frozen=True)
class DataValidationReport:
    """Summary of data checks performed before modeling."""

    source_path: str
    loaded_rows: int
    cleaned_rows: int
    duplicate_label_message_rows_removed: int
    missing_message_rows: int
    conflicting_message_count: int
    conflicting_messages: tuple[str, ...]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a two-column label/message DataFrame from common UCI shapes."""
    if {"label", "message"}.issubset(df.columns):
        return df.loc[:, ["label", "message"]].copy()

    lower_to_original = {str(col).strip().lower(): col for col in df.columns}
    label_candidates = ["label", "class", "v1", "target"]
    message_candidates = ["message", "sms", "text", "v2"]

    label_col = next((lower_to_original[c] for c in label_candidates if c in lower_to_original), None)
    message_col = next((lower_to_original[c] for c in message_candidates if c in lower_to_original), None)

    if label_col is not None and message_col is not None:
        return df.loc[:, [label_col, message_col]].rename(
            columns={label_col: "label", message_col: "message"}
        )

    if df.shape[1] >= 2:
        return df.iloc[:, :2].copy().set_axis(["label", "message"], axis=1)

    raise ValueError("Expected a DataFrame containing label and message columns.")


def find_conflicting_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Identify exact message text that appears with more than one label."""
    required = {"label", "message"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    grouped = (
        df.groupby("message", dropna=False)
        .agg(
            label_count=("label", "nunique"),
            labels=("label", lambda values: tuple(sorted(set(values)))),
            rows=("label", "size"),
        )
        .reset_index()
    )
    conflicts = grouped[grouped["label_count"] > 1].copy()
    return conflicts.loc[:, ["message", "labels", "rows"]].reset_index(drop=True)


def validate_sms_dataframe(
    df: pd.DataFrame,
    *,
    source_path: str = "<memory>",
    drop_exact_duplicates: bool = True,
) -> tuple[pd.DataFrame, DataValidationReport]:
    """Validate labels/messages and remove exact duplicate label-message rows."""
    required = {"label", "message"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required column(s): {sorted(missing_columns)}")

    normalized = df.loc[:, ["label", "message"]].copy()
    loaded_rows = len(normalized)

    unexpected_labels = sorted(set(normalized["label"].dropna()) - set(LABELS))
    if normalized["label"].isna().any() or unexpected_labels:
        raise ValueError(
            "Unexpected label values. Expected only "
            f"{list(LABELS)}; found {unexpected_labels or ['<missing>']}."
        )

    missing_messages = normalized["message"].isna()
    empty_messages = normalized["message"].astype("string").str.strip().eq("")
    missing_message_rows = int((missing_messages | empty_messages).sum())
    if missing_message_rows:
        raise ValueError(f"Found {missing_message_rows} missing or empty message row(s).")

    conflicts = find_conflicting_messages(normalized)

    if drop_exact_duplicates:
        cleaned = normalized.drop_duplicates(subset=["label", "message"], keep="first")
    else:
        cleaned = normalized.copy()

    duplicate_rows_removed = loaded_rows - len(cleaned)
    cleaned = cleaned.reset_index(drop=True)

    report = DataValidationReport(
        source_path=source_path,
        loaded_rows=loaded_rows,
        cleaned_rows=len(cleaned),
        duplicate_label_message_rows_removed=duplicate_rows_removed,
        missing_message_rows=missing_message_rows,
        conflicting_message_count=len(conflicts),
        conflicting_messages=tuple(conflicts["message"].tolist()),
    )
    return cleaned, report


def _select_ucimlrepo_frame(dataset: object) -> pd.DataFrame:
    data = getattr(dataset, "data", None)
    if data is None:
        raise ValueError("ucimlrepo did not return a data object.")

    original = getattr(data, "original", None)
    if isinstance(original, pd.DataFrame) and not original.empty:
        return _normalize_columns(original)

    targets = getattr(data, "targets", None)
    features = getattr(data, "features", None)
    if isinstance(targets, pd.DataFrame) and isinstance(features, pd.DataFrame):
        combined = pd.concat([targets.iloc[:, :1], features.iloc[:, :1]], axis=1)
        return _normalize_columns(combined)

    raise ValueError("Could not locate label/message data in ucimlrepo response.")


def _download_from_uci_archive(raw_path: Path) -> Path:
    with urlopen(UCI_ARCHIVE_URL, timeout=60) as response:
        archive_bytes = response.read()

    with ZipFile(BytesIO(archive_bytes)) as archive:
        matching_names = [
            name
            for name in archive.namelist()
            if Path(name).name == "SMSSpamCollection"
        ]
        if not matching_names:
            raise RuntimeError("UCI archive did not contain SMSSpamCollection.")
        raw_bytes = archive.read(matching_names[0])

    raw_path.write_bytes(raw_bytes)
    return raw_path


def download_sms_spam_collection(raw_path: str | Path = DEFAULT_RAW_PATH) -> Path:
    """Download UCI dataset ID 228 and write the local raw file.

    The helper first attempts `ucimlrepo.fetch_ucirepo(id=228)`. UCI currently
    lists this dataset in `ucimlrepo` but may not expose it for direct import,
    so the helper falls back to UCI's official static archive for the same ID.
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "ucimlrepo is required to retrieve the dataset. Install the project with "
            "`python -m pip install -e .` and rerun this command."
        ) from exc

    raw_path = Path(raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        dataset = fetch_ucirepo(id=UCI_DATASET_ID)
        normalized = _select_ucimlrepo_frame(dataset)
        normalized, _ = validate_sms_dataframe(
            normalized,
            source_path=f"ucimlrepo:{UCI_DATASET_ID}",
            drop_exact_duplicates=False,
        )
        normalized.to_csv(raw_path, sep="\t", header=False, index=False)
        return raw_path
    except Exception:
        return _download_from_uci_archive(raw_path)


def load_sms_spam_collection(
    raw_path: str | Path = DEFAULT_RAW_PATH,
    *,
    download_if_missing: bool = False,
    return_report: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, DataValidationReport]:
    """Load, validate, and de-duplicate the SMS Spam Collection.

    The raw file is expected to be a tab-separated file with label and message
    fields. Message text is preserved exactly for feature extraction.
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        if download_if_missing:
            raw_path = download_sms_spam_collection(raw_path)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {raw_path}. Run "
                "`python -c \"from spam_classifier.data import "
                "download_sms_spam_collection; download_sms_spam_collection()\"`."
            )

    df = pd.read_csv(
        raw_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        dtype={"label": "string", "message": "string"},
        keep_default_na=False,
    )
    cleaned, report = validate_sms_dataframe(df, source_path=str(raw_path))
    if return_report:
        return cleaned, report
    return cleaned


def label_counts(df: pd.DataFrame, labels: Iterable[str] = LABELS) -> pd.DataFrame:
    """Return counts and percentages in a stable label order."""
    counts = df["label"].value_counts().reindex(labels, fill_value=0)
    result = counts.rename_axis("label").reset_index(name="count")
    result["percent"] = result["count"] / len(df)
    return result
