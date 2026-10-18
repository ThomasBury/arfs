"""C03: default classification folds preserve classes; custom folds take priority."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    RepeatedKFold,
    RepeatedStratifiedKFold,
)

from arfs.feature_selection import allrelevant as a


@pytest.fixture
def training_folds(monkeypatch):
    seen = []

    def train(frame, target, weight, validation, validation_target, *args, **kwargs):
        seen.append(
            (
                frame.index.to_numpy(),
                validation.index.to_numpy(),
                target.to_numpy(),
                validation_target.to_numpy(),
            )
        )
        return SimpleNamespace(best_iteration=1), None, 1

    monkeypatch.setattr(a, "_train_lgb_model", train)
    monkeypatch.setattr(
        a,
        "_compute_importance",
        lambda frame, *args: list(zip(frame.columns, np.ones(frame.shape[1]))),
    )
    return seen


@pytest.mark.parametrize(
    "objective",
    [
        "binary",
        "binary_logloss",
        "softmax",
        "multi_logloss",
        "multiclass",
        "multiclassova",
        "multiclass_ova",
        "ova",
        "ovr",
    ],
)
@pytest.mark.parametrize("counts", [(50, 50), (95, 5), (80, 15, 5)])
def test_recognized_classification_defaults_match_stratified_reference(
    objective, counts, training_folds
):
    y = pd.Series(np.repeat(np.arange(len(counts)), counts))
    X = pd.DataFrame({"x": np.arange(len(y))})
    expected = list(
        RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=81).split(X, y)
    )
    a.GrootCV(objective=objective, n_folds=5, n_iter=2, random_state=81).fit(X, y)
    assert len(training_folds) == 10
    for (train, valid, train_y, valid_y), (expected_train, expected_valid) in zip(
        training_folds, expected
    ):
        assert_array_equal(train, expected_train)
        assert_array_equal(valid, expected_valid)
        assert set(train_y) == set(valid_y) == set(range(len(counts)))


def test_classification_counts_ignore_unused_target_categories(training_folds):
    y = pd.Series(pd.Categorical([0, 1] * 10, categories=[0, 1, 2]))
    X = pd.DataFrame({"x": np.arange(len(y))})
    a.GrootCV(objective="binary", n_folds=5, n_iter=1).fit(X, y)
    assert len(training_folds) == 5
    assert all(set(valid_y) == {0, 1} for _, _, _, valid_y in training_folds)


def test_custom_splitter_can_require_data_to_count_splits(training_folds):
    X = pd.DataFrame({"x": np.arange(6)})
    a.GrootCV(objective="l2", folds=LeaveOneOut()).fit(X, pd.Series(np.arange(6.0)))
    assert len(training_folds) == 6
    assert all(len(valid) == 1 for _, valid, *_ in training_folds)


def test_rare_class_default_folds_reach_native_training(monkeypatch):
    y = pd.Series([0] * 95 + [1] * 5)
    X = pd.DataFrame({"signal": y + np.random.RandomState(5).normal(0, 0.1, len(y))})
    train = a._train_lgb_model
    seen = []

    def checked_train(
        frame, target, weight, validation, validation_target, *args, **kwargs
    ):
        seen.append(
            (
                target.value_counts().to_dict(),
                validation_target.value_counts().to_dict(),
            )
        )
        return train(
            frame, target, weight, validation, validation_target, *args, **kwargs
        )

    monkeypatch.setattr(a, "_train_lgb_model", checked_train)
    selector = a.GrootCV(
        objective="binary",
        n_folds=5,
        n_iter=1,
        n_jobs=1,
        lgbm_params={"min_data_in_leaf": 2, "seed": 5},
    ).fit(X, y)
    assert seen == [({0: 76, 1: 4}, {0: 19, 1: 1})] * 5
    assert np.isfinite(selector.cv_df.filter(like="fscore")).all().all()


@pytest.mark.parametrize(
    "objective,params,classification",
    [
        (None, {"objective": "binary"}, True),
        (None, {"objective": "multiclass"}, True),
        ("binary", {"objective": "l2"}, True),
        ("l2", {"objective": "binary"}, False),
        ("regression", None, False),
    ],
)
def test_resolved_objective_controls_default_fold_type(
    objective, params, classification, training_folds
):
    y = pd.Series([0] * 25 + [1] * 5)
    X = pd.DataFrame({"x": np.arange(len(y))})
    splitter = RepeatedStratifiedKFold if classification else RepeatedKFold
    expected = list(splitter(n_splits=5, n_repeats=2, random_state=81).split(X, y))
    a.GrootCV(
        objective=objective, lgbm_params=params, n_folds=5, n_iter=2, random_state=81
    ).fit(X, y)
    assert len(training_folds) == len(expected)
    for (train, valid, *_), (expected_train, expected_valid) in zip(
        training_folds, expected
    ):
        assert_array_equal(train, expected_train)
        assert_array_equal(valid, expected_valid)


@pytest.mark.parametrize("kind", ["list", "iterator", "generator", "splitter"])
def test_empty_custom_folds_fail_before_training(kind, training_folds):
    folds = {
        "list": [],
        "iterator": iter(()),
        "generator": (pair for pair in ()),
        "splitter": SimpleNamespace(split=lambda X, y: iter(())),
    }[kind]
    X = pd.DataFrame({"x": np.arange(10)})
    y = pd.Series([0, 1] * 5)
    with pytest.raises(ValueError, match="folds.*at least one.*new iterator"):
        a.GrootCV(objective="binary", folds=folds).fit(X, y)
    assert not training_folds


@pytest.mark.parametrize("counts", [(20, 4), (20, 10, 1), (20,)])
def test_infeasible_default_stratification_fails_before_training(
    counts, training_folds
):
    y = pd.Series(np.repeat(np.arange(len(counts)), counts))
    X = pd.DataFrame({"x": np.arange(len(y))})
    with pytest.raises(ValueError, match="stratification.*n_folds.*folds"):
        a.GrootCV(objective="binary", n_folds=5, n_iter=1).fit(X, y)
    assert not training_folds


@pytest.mark.parametrize("kind", ["splitter", "list", "iterator", "generator"])
def test_custom_folds_override_infeasible_defaults(kind, training_folds):
    y = pd.Series([0] * 19 + [1])
    X = pd.DataFrame({"x": np.arange(len(y))})
    splitter = KFold(n_splits=2, shuffle=True, random_state=7)
    expected = list(splitter.split(X, y))
    folds = {
        "splitter": splitter,
        "list": expected,
        "iterator": iter(expected),
        "generator": (pair for pair in expected),
    }[kind]
    selector = a.GrootCV(objective="binary", n_folds=5, n_iter=3, folds=folds)
    selector.fit(X, y)
    assert selector.folds is folds
    assert len(training_folds) == 2
    for (train, valid, *_), (expected_train, expected_valid) in zip(
        training_folds, expected
    ):
        assert_array_equal(train, expected_train)
        assert_array_equal(valid, expected_valid)

    if kind in ("iterator", "generator"):
        with pytest.raises(ValueError, match="folds.*new iterator"):
            selector.fit(X, y)
        assert len(training_folds) == 2
    else:
        selector.fit(X, y)
        assert len(training_folds) == 4
        for (train, valid, *_), (expected_train, expected_valid) in zip(
            training_folds[2:], expected
        ):
            assert_array_equal(train, expected_train)
            assert_array_equal(valid, expected_valid)
    assert selector.folds is folds
