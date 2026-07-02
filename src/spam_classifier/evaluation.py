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


def _negative_label() -> str:
    negative_labels = [label for label in LABELS if label != POS_LABEL]
    if len(negative_labels) != 1:
        raise ValueError("Threshold metrics require exactly one non-spam label.")
    return negative_labels[0]


def _one_dimensional_scores(scores) -> np.ndarray:
    score_array = np.asarray(scores, dtype=float)
    if score_array.ndim != 1:
        raise ValueError("Expected one-dimensional spam-oriented scores.")
    if not np.isfinite(score_array).all():
        raise ValueError("Decision scores must be finite.")
    return score_array


def _positive_index(model) -> int:
    classes = list(getattr(model, "classes_", []))
    if POS_LABEL not in classes:
        raise ValueError(f"Model classes do not include positive label {POS_LABEL!r}.")
    return classes.index(POS_LABEL)


def orient_decision_scores(scores, classes) -> np.ndarray:
    """Return one-dimensional decision scores where larger values favor spam."""
    score_array = np.asarray(scores, dtype=float)
    class_list = list(classes)
    if POS_LABEL not in class_list:
        raise ValueError(f"Classes do not include positive label {POS_LABEL!r}.")

    if score_array.ndim == 1:
        if len(class_list) != 2:
            raise ValueError("One-dimensional decision scores require two classes.")
        if class_list[1] == POS_LABEL:
            return score_array
        if class_list[0] == POS_LABEL:
            return -score_array
        raise ValueError(f"Could not orient scores for classes {class_list!r}.")

    if score_array.ndim == 2:
        if score_array.shape[1] != len(class_list):
            raise ValueError("Score columns must match the number of classes.")
        return score_array[:, class_list.index(POS_LABEL)]

    raise ValueError("Expected one- or two-dimensional decision scores.")


def prediction_scores(model, X) -> np.ndarray | None:
    """Return spam-oriented probabilities or decision scores when available."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        return probabilities[:, _positive_index(model)]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        classes = list(getattr(model, "classes_", []))
        return orient_decision_scores(scores, classes)

    return None


def predict_labels_at_threshold(scores, threshold: float = 0.0) -> np.ndarray:
    """Convert spam-oriented scores into labels at a fixed decision threshold.

    Scores strictly greater than the threshold are labeled as spam. With a
    threshold of 0, this matches the default binary `LinearSVC.predict` rule.
    """
    score_array = _one_dimensional_scores(scores)
    return np.where(score_array > threshold, POS_LABEL, _negative_label())


def threshold_metrics(y_true, scores, threshold: float) -> dict[str, float | int]:
    """Calculate spam metrics and error counts for one decision threshold."""
    y_true_array = np.asarray(y_true)
    y_pred = predict_labels_at_threshold(scores, threshold)

    return {
        "threshold": float(threshold),
        "spam_precision": precision_score(
            y_true_array, y_pred, pos_label=POS_LABEL, zero_division=0
        ),
        "spam_recall": recall_score(
            y_true_array, y_pred, pos_label=POS_LABEL, zero_division=0
        ),
        "spam_f1": f1_score(y_true_array, y_pred, pos_label=POS_LABEL, zero_division=0),
        "false_positives": int(
            ((y_true_array != POS_LABEL) & (y_pred == POS_LABEL)).sum()
        ),
        "false_negatives": int(
            ((y_true_array == POS_LABEL) & (y_pred != POS_LABEL)).sum()
        ),
    }


def threshold_comparison_table(y_true, scores, thresholds) -> pd.DataFrame:
    """Compare spam metrics across candidate decision thresholds."""
    threshold_array = np.unique(np.asarray(list(thresholds), dtype=float))
    if threshold_array.size == 0:
        raise ValueError("At least one candidate threshold is required.")
    if not np.isfinite(threshold_array).all():
        raise ValueError("Candidate thresholds must be finite.")

    rows = [
        threshold_metrics(y_true, scores, threshold)
        for threshold in np.sort(threshold_array)
    ]
    return pd.DataFrame(rows)


def select_best_threshold(y_true, scores, thresholds) -> pd.Series:
    """Select the threshold with best spam F1, recall, then closeness to 0."""
    comparison = threshold_comparison_table(y_true, scores, thresholds)
    ranked = (
        comparison.assign(_distance_to_default=comparison["threshold"].abs())
        .sort_values(
            ["spam_f1", "spam_recall", "_distance_to_default", "threshold"],
            ascending=[False, False, True, True],
        )
        .drop(columns="_distance_to_default")
    )
    return ranked.iloc[0].copy()


def candidate_thresholds_from_scores(
    scores,
    *,
    include_default: bool = True,
) -> np.ndarray:
    """Return candidate thresholds covering each distinct score-based operating point."""
    score_array = _one_dimensional_scores(scores)
    if score_array.size == 0:
        raise ValueError("At least one decision score is required.")

    unique_scores = np.unique(score_array)
    if unique_scores.size == 1:
        thresholds = np.array(
            [
                np.nextafter(unique_scores[0], -np.inf),
                np.nextafter(unique_scores[0], np.inf),
            ]
        )
    else:
        midpoints = unique_scores[:-1] / 2 + unique_scores[1:] / 2
        thresholds = np.concatenate(
            [
                [np.nextafter(unique_scores[0], -np.inf)],
                midpoints,
                [np.nextafter(unique_scores[-1], np.inf)],
            ]
        )

    if include_default:
        thresholds = np.concatenate([thresholds, [0.0]])
    return np.unique(thresholds.astype(float))


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
