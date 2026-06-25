"""Evaluation helpers that consistently treat spam as the positive class."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from spam_classifier.data import LABELS, POS_LABEL


def _positive_index(model) -> int:
    classes = list(getattr(model, "classes_", []))
    if POS_LABEL not in classes:
        raise ValueError(f"Model classes do not include positive label {POS_LABEL!r}.")
    return classes.index(POS_LABEL)


def prediction_scores(model, X) -> np.ndarray | None:
    """Return spam-oriented probabilities or decision scores when available."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        return probabilities[:, _positive_index(model)]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        classes = list(getattr(model, "classes_", []))
        if np.ndim(scores) == 1:
            if len(classes) == 2 and classes[1] != POS_LABEL:
                return -scores
            return scores
        return scores[:, _positive_index(model)]

    return None


def evaluate_classifier(model, X, y_true) -> dict[str, object]:
    """Compute class-sensitive metrics for a fitted classifier."""
    y_pred = model.predict(X)
    metrics = {
        "spam_precision": precision_score(
            y_true, y_pred, pos_label=POS_LABEL, zero_division=0
        ),
        "spam_recall": recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "spam_f1": f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)),
    }

    scores = prediction_scores(model, X)
    if scores is not None:
        metrics["average_precision"] = average_precision_score(
            np.asarray(y_true) == POS_LABEL,
            scores,
        )
    else:
        metrics["average_precision"] = np.nan

    return metrics


def confusion_matrix_frame(matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in LABELS],
        columns=[f"predicted_{label}" for label in LABELS],
    )


def error_examples(model, X, y_true, *, limit_per_type: int = 5) -> pd.DataFrame:
    """Return a balanced sample of false positives and false negatives."""
    y_pred = model.predict(X)
    scores = prediction_scores(model, X)
    result = pd.DataFrame(
        {
            "actual_label": list(y_true),
            "predicted_label": list(y_pred),
            "message": list(X),
        }
    )
    if scores is not None:
        result["score"] = scores

    false_positives = result[
        (result["actual_label"] != POS_LABEL)
        & (result["predicted_label"] == POS_LABEL)
    ].copy()
    false_positives.insert(0, "error_type", "false_positive")

    false_negatives = result[
        (result["actual_label"] == POS_LABEL)
        & (result["predicted_label"] != POS_LABEL)
    ].copy()
    false_negatives.insert(0, "error_type", "false_negative")

    if "score" in false_positives:
        false_positives["score"] = false_positives["score"].round(4)
        false_positives = false_positives.sort_values("score", ascending=False)
    if "score" in false_negatives:
        false_negatives["score"] = false_negatives["score"].round(4)
        false_negatives = false_negatives.sort_values("score", ascending=True)

    return pd.concat(
        [
            false_positives.head(limit_per_type),
            false_negatives.head(limit_per_type),
        ],
        ignore_index=True,
    )
