import pandas as pd

from spam_classifier.data import LABELS
from spam_classifier.evaluation import error_examples, evaluate_classifier
from spam_classifier.modeling import make_model_specs


def tiny_sms_fixture():
    return pd.DataFrame(
        {
            "label": ["ham", "spam", "ham", "spam", "ham", "spam"],
            "message": [
                "are we still meeting for lunch today",
                "win cash prize by texting claim now",
                "please call me when you get home",
                "urgent prize winner call now",
                "can you pick up milk later",
                "claim your free voucher today",
            ],
        }
    )


def test_pipeline_construction_includes_tfidf_and_classifier():
    specs = make_model_specs()

    assert {spec.name for spec in specs} == {
        "DummyClassifier",
        "MultinomialNB",
        "LogisticRegression",
        "LinearSVC",
    }
    for spec in specs:
        assert "tfidf" in spec.pipeline.named_steps
        assert "classifier" in spec.pipeline.named_steps


def test_pipeline_fits_and_predicts_expected_labels():
    df = tiny_sms_fixture()
    spec = next(spec for spec in make_model_specs() if spec.name == "MultinomialNB")
    model = spec.pipeline

    model.fit(df["message"], df["label"])
    predictions = model.predict(df["message"])

    assert len(predictions) == len(df)
    assert set(predictions).issubset(set(LABELS))


def test_metric_helper_treats_spam_as_positive():
    df = tiny_sms_fixture()
    spec = next(spec for spec in make_model_specs() if spec.name == "MultinomialNB")
    model = spec.pipeline.fit(df["message"], df["label"])

    metrics = evaluate_classifier(model, df["message"], df["label"])

    assert metrics["spam_precision"] == 1.0
    assert metrics["spam_recall"] == 1.0
    assert metrics["spam_f1"] == 1.0
    assert metrics["confusion_matrix"].shape == (2, 2)


class FixedScoreModel:
    classes_ = ["ham", "spam"]

    def predict(self, X):
        return ["spam", "ham", "ham", "spam", "spam", "ham"][: len(X)]

    def decision_function(self, X):
        return [2.0, -1.5, -2.0, 1.8, 0.5, -0.2][: len(X)]


class FixedNoScoreModel:
    def predict(self, X):
        return ["spam", "ham", "ham", "spam", "spam", "ham"][: len(X)]


def error_fixture():
    return pd.DataFrame(
        {
            "label": ["ham", "spam", "ham", "spam", "ham", "spam"],
            "message": [
                "normal message predicted spam",
                "spam message predicted ham",
                "normal message predicted ham",
                "spam message predicted spam",
                "second normal message predicted spam",
                "second spam message predicted ham",
            ],
        }
    )


def test_error_examples_returns_both_error_types_with_correct_labels():
    df = error_fixture()

    errors = error_examples(
        FixedScoreModel(), df["message"], df["label"], limit_per_type=5
    )

    assert errors["error_type"].tolist() == [
        "false_positive",
        "false_positive",
        "false_negative",
        "false_negative",
    ]
    assert set(errors["error_type"]) == {"false_positive", "false_negative"}
    assert (errors.loc[errors["error_type"] == "false_positive", "actual_label"] == "ham").all()
    assert (errors.loc[errors["error_type"] == "false_positive", "predicted_label"] == "spam").all()
    assert (errors.loc[errors["error_type"] == "false_negative", "actual_label"] == "spam").all()
    assert (errors.loc[errors["error_type"] == "false_negative", "predicted_label"] == "ham").all()


def test_error_examples_enforces_limit_per_type_and_score_sorting():
    df = error_fixture()

    errors = error_examples(
        FixedScoreModel(), df["message"], df["label"], limit_per_type=1
    )

    assert errors["error_type"].tolist() == ["false_positive", "false_negative"]
    assert errors["score"].tolist() == [2.0, -1.5]


def test_error_examples_works_without_scores():
    df = error_fixture()

    errors = error_examples(
        FixedNoScoreModel(), df["message"], df["label"], limit_per_type=1
    )

    assert errors["error_type"].tolist() == ["false_positive", "false_negative"]
    assert "score" not in errors.columns
