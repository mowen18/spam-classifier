"""Model construction and selection utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from spam_classifier.data import POS_LABEL

RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelSpec:
    name: str
    pipeline: Pipeline
    param_grid: dict[str, list[object]]


def make_text_pipeline(classifier: object) -> Pipeline:
    """Create a leakage-safe text pipeline with TF-IDF inside the estimator."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            ("classifier", classifier),
        ]
    )


def make_train_test_split(df: pd.DataFrame):
    """Create the single stratified train/test split used by the project."""
    return train_test_split(
        df["message"],
        df["label"],
        test_size=0.20,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def make_scoring() -> dict[str, object]:
    return {
        "spam_f1": make_scorer(f1_score, pos_label=POS_LABEL, zero_division=0),
        "spam_precision": make_scorer(
            precision_score, pos_label=POS_LABEL, zero_division=0
        ),
        "spam_recall": make_scorer(recall_score, pos_label=POS_LABEL, zero_division=0),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
    }


def make_model_specs() -> list[ModelSpec]:
    """Return the focused model set and modest hyperparameter grids."""
    vectorizer_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
    }

    return [
        ModelSpec(
            name="DummyClassifier",
            pipeline=make_text_pipeline(
                DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
            ),
            param_grid={
                "tfidf__ngram_range": [(1, 1)],
                "tfidf__min_df": [1],
            },
        ),
        ModelSpec(
            name="MultinomialNB",
            pipeline=make_text_pipeline(MultinomialNB()),
            param_grid={
                **vectorizer_grid,
                "tfidf__sublinear_tf": [True],
                "tfidf__stop_words": [None],
                "classifier__alpha": [0.1, 0.5, 1.0],
            },
        ),
        ModelSpec(
            name="LogisticRegression",
            pipeline=make_text_pipeline(
                LogisticRegression(
                    solver="liblinear",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                )
            ),
            param_grid={
                **vectorizer_grid,
                "tfidf__sublinear_tf": [True],
                "tfidf__stop_words": [None, "english"],
                "classifier__C": [0.5, 1.0, 2.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        ModelSpec(
            name="LinearSVC",
            pipeline=make_text_pipeline(
                LinearSVC(max_iter=5000, random_state=RANDOM_STATE)
            ),
            param_grid={
                **vectorizer_grid,
                "tfidf__sublinear_tf": [True],
                "tfidf__stop_words": [None, "english"],
                "classifier__C": [0.5, 1.0, 2.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
    ]


def run_model_search(X_train, y_train, *, n_jobs: int = 1):
    """Fit GridSearchCV for each candidate using spam F1 as the refit metric."""
    rows = []
    searches = {}
    scoring = make_scoring()
    cv = make_cv()

    for spec in make_model_specs():
        search = GridSearchCV(
            estimator=spec.pipeline,
            param_grid=spec.param_grid,
            scoring=scoring,
            refit="spam_f1",
            cv=cv,
            n_jobs=n_jobs,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        searches[spec.name] = search
        best_index = search.best_index_
        rows.append(
            {
                "model": spec.name,
                "mean_cv_spam_f1": search.cv_results_["mean_test_spam_f1"][best_index],
                "std_cv_spam_f1": search.cv_results_["std_test_spam_f1"][best_index],
                "mean_cv_spam_precision": search.cv_results_[
                    "mean_test_spam_precision"
                ][best_index],
                "mean_cv_spam_recall": search.cv_results_["mean_test_spam_recall"][
                    best_index
                ],
                "mean_cv_balanced_accuracy": search.cv_results_[
                    "mean_test_balanced_accuracy"
                ][best_index],
                "best_params": search.best_params_,
            }
        )

    comparison = (
        pd.DataFrame(rows)
        .sort_values(["mean_cv_spam_f1", "mean_cv_spam_recall"], ascending=False)
        .reset_index(drop=True)
    )
    return comparison, searches
