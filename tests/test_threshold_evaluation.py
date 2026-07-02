import numpy as np
import pytest
from sklearn.svm import LinearSVC

from spam_classifier.evaluation import (
    orient_decision_scores,
    predict_labels_at_threshold,
    prediction_scores,
    select_best_threshold,
    threshold_metrics,
)
from spam_classifier.modeling import RANDOM_STATE, make_text_pipeline


def test_lower_threshold_predicts_at_least_as_many_spam_messages():
    scores = np.array([-1.0, -0.2, 0.0, 0.3, 1.0])

    lower_threshold_predictions = predict_labels_at_threshold(scores, threshold=-0.5)
    higher_threshold_predictions = predict_labels_at_threshold(scores, threshold=0.5)

    assert (lower_threshold_predictions == "spam").sum() >= (
        higher_threshold_predictions == "spam"
    ).sum()


def test_prediction_behavior_at_exact_threshold():
    scores = np.array([-0.1, 0.0, 0.1])

    predictions = predict_labels_at_threshold(scores, threshold=0.0)

    assert predictions.tolist() == ["ham", "ham", "spam"]


def test_threshold_metrics_count_false_positives_and_false_negatives():
    y_true = np.array(["ham", "spam", "spam", "ham"])
    scores = np.array([0.4, 0.3, -0.1, -0.2])

    metrics = threshold_metrics(y_true, scores, threshold=0.0)

    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1


def test_threshold_metrics_treat_spam_as_the_positive_class():
    y_true = np.array(["ham", "spam", "spam", "ham"])
    scores = np.array([0.4, 0.3, -0.1, -0.2])

    metrics = threshold_metrics(y_true, scores, threshold=0.0)

    assert metrics["spam_precision"] == pytest.approx(0.5)
    assert metrics["spam_recall"] == pytest.approx(0.5)
    assert metrics["spam_f1"] == pytest.approx(0.5)


def test_orient_decision_scores_makes_larger_values_favor_spam():
    raw_scores = np.array([2.0, -1.0])

    oriented = orient_decision_scores(raw_scores, classes=["spam", "ham"])

    assert oriented.tolist() == [-2.0, 1.0]


def test_select_best_threshold_chooses_highest_spam_f1():
    y_true = np.array(["ham", "spam", "spam", "ham"])
    scores = np.array([-0.8, -0.2, 0.1, 0.5])

    selected = select_best_threshold(
        y_true,
        scores,
        thresholds=[-0.3, 0.0, 0.6],
    )

    assert selected["threshold"] == pytest.approx(-0.3)
    assert selected["spam_f1"] == pytest.approx(0.8)


def test_select_best_threshold_prefers_recall_when_f1_ties():
    y_true = np.array(["spam", "spam", "ham", "ham", "spam", "spam"])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

    selected = select_best_threshold(
        y_true,
        scores,
        thresholds=[0.75, 0.45],
    )

    assert selected["threshold"] == pytest.approx(0.45)
    assert selected["spam_recall"] == pytest.approx(0.75)


def test_select_best_threshold_prefers_threshold_closest_to_default_when_still_tied():
    y_true = np.array(["ham", "spam"])
    scores = np.array([-0.2, 0.2])

    selected = select_best_threshold(
        y_true,
        scores,
        thresholds=[-0.05, 0.15],
    )

    assert selected["threshold"] == pytest.approx(-0.05)


def test_default_threshold_matches_linear_svc_predict():
    messages = np.array(
        [
            "see you at lunch",
            "call me after work",
            "family dinner tonight",
            "project meeting tomorrow",
            "win cash prize now",
            "claim free voucher",
            "urgent lottery winner",
            "free cash reward claim",
        ]
    )
    labels = np.array(["ham", "ham", "ham", "ham", "spam", "spam", "spam", "spam"])
    model = make_text_pipeline(
        LinearSVC(max_iter=5000, random_state=RANDOM_STATE)
    ).fit(messages, labels)

    scores = prediction_scores(model, messages)
    threshold_predictions = predict_labels_at_threshold(scores, threshold=0.0)

    assert np.array_equal(threshold_predictions, model.predict(messages))
