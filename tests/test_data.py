import pandas as pd
import pytest

from spam_classifier.data import (
    find_conflicting_messages,
    label_counts,
    validate_sms_dataframe,
)


def test_required_columns_are_enforced():
    df = pd.DataFrame({"label": ["ham"], "text": ["hello"]})

    with pytest.raises(ValueError, match="Missing required"):
        validate_sms_dataframe(df)


def test_valid_labels_are_enforced():
    df = pd.DataFrame({"label": ["ham", "junk"], "message": ["hello", "win"]})

    with pytest.raises(ValueError, match="Unexpected label"):
        validate_sms_dataframe(df)


def test_missing_message_behavior_is_explicit():
    df = pd.DataFrame({"label": ["ham", "spam"], "message": ["hello", None]})

    with pytest.raises(ValueError, match="missing or empty message"):
        validate_sms_dataframe(df)


def test_exact_duplicate_rows_are_removed():
    df = pd.DataFrame(
        {
            "label": ["ham", "ham", "spam"],
            "message": ["see you soon", "see you soon", "claim prize now"],
        }
    )

    cleaned, report = validate_sms_dataframe(df)

    assert len(cleaned) == 2
    assert report.loaded_rows == 3
    assert report.cleaned_rows == 2
    assert report.duplicate_label_message_rows_removed == 1


def test_conflicting_label_detection():
    df = pd.DataFrame(
        {
            "label": ["ham", "spam", "spam"],
            "message": ["same text", "same text", "different text"],
        }
    )

    conflicts = find_conflicting_messages(df)

    assert conflicts.shape[0] == 1
    assert conflicts.loc[0, "message"] == "same text"
    assert conflicts.loc[0, "labels"] == ("ham", "spam")


def test_label_counts_uses_stable_label_order():
    df = pd.DataFrame(
        {
            "label": ["spam", "ham", "ham"],
            "message": ["win", "hello", "later"],
        }
    )

    counts = label_counts(df)

    assert counts["label"].tolist() == ["ham", "spam"]
    assert counts["count"].tolist() == [2, 1]

