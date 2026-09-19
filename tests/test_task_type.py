"""Target-column selection / task-type detection tests (issue #5).

Pure-logic tests against `train.task_type` -- no Django test client needed,
mirroring how `automl_core.validation` is tested independent of any view.
"""
import pandas as pd
import pytest

from train.task_type import TargetColumnError, analyze_target_column, detect_task_type


def _series(values):
    return pd.Series(values)


# --- detect_task_type: low-level heuristic -----------------------------


def test_binary_numeric_labels_detected_as_classification():
    task_type, warning = detect_task_type(_series([0, 1, 0, 1, 1, 0, 1, 0]))
    assert task_type == "classification"
    assert warning is None


def test_multiclass_numeric_labels_detected_as_classification():
    task_type, warning = detect_task_type(_series([0, 1, 2, 1, 0, 2, 2, 1, 0]))
    assert task_type == "classification"
    assert warning is None


def test_all_numeric_but_actually_categorical_is_not_regression():
    # Regression heuristic bug this guards against: "numeric -> regression"
    # would silently misclassify a small integer-coded label column.
    task_type, _ = detect_task_type(_series([1, 2, 3, 1, 2, 3, 1, 2, 3, 1]))
    assert task_type == "classification"


def test_continuous_numeric_column_detected_as_regression():
    task_type, warning = detect_task_type(
        _series([1.23, 4.56, 7.89, 2.34, 5.67, 8.91, 3.45, 6.78, 9.01, 0.12])
    )
    assert task_type == "regression"
    assert warning is None


def test_high_unique_count_numeric_detected_as_regression():
    task_type, _ = detect_task_type(_series(list(range(50))))
    assert task_type == "regression"


def test_bool_column_detected_as_classification():
    task_type, _ = detect_task_type(_series([True, False, True, True, False]))
    assert task_type == "classification"


def test_low_cardinality_string_column_detected_as_classification():
    labels = ["cat", "dog", "cat", "bird", "dog", "cat", "bird", "dog", "cat", "bird"]
    task_type, warning = detect_task_type(_series(labels))
    assert task_type == "classification"
    assert warning is None


def test_high_cardinality_string_column_is_ambiguous():
    # Looks like an identifier/free-text column (9 distinct values across 10
    # rows) -- auto-detection must refuse to guess rather than returning a
    # 9-class "classification" target.
    labels = [f"user_{i}" for i in range(9)] + ["user_0"]
    task_type, warning = detect_task_type(_series(labels))
    assert task_type is None
    assert warning is not None
    assert "identifier" in warning or "free-text" in warning


# --- analyze_target_column: full validation + detection -----------------


def test_analyze_infers_classification_for_binary_target():
    df = pd.DataFrame({"label": [0, 1, 0, 1, 1, 0], "x": range(6)})
    result = analyze_target_column(df, "label")
    assert result.detected_task_type == "classification"
    assert result.effective_task_type == "classification"
    assert result.overridden is False
    assert result.n_unique == 2


def test_analyze_infers_classification_for_multiclass_target():
    df = pd.DataFrame({"label": [0, 1, 2, 1, 0, 2, 2], "x": range(7)})
    result = analyze_target_column(df, "label")
    assert result.effective_task_type == "classification"
    assert result.n_unique == 3


def test_analyze_infers_regression_for_continuous_target():
    df = pd.DataFrame({"price": [10.5, 20.1, 30.7, 40.2, 50.9, 15.3, 25.8], "x": range(7)})
    result = analyze_target_column(df, "price")
    assert result.detected_task_type == "regression"
    assert result.effective_task_type == "regression"


def test_analyze_allows_override_of_detected_task_type():
    # A low-cardinality numeric column auto-detects as classification; an
    # explicit override must still win for an otherwise-valid column.
    df = pd.DataFrame({"label": [0, 1, 0, 1, 1, 0], "x": range(6)})
    result = analyze_target_column(df, "label", override_task_type="regression")
    assert result.detected_task_type == "classification"
    assert result.effective_task_type == "regression"
    assert result.overridden is True


def test_analyze_high_cardinality_string_requires_override():
    labels = [f"user_{i}" for i in range(9)] + ["user_0"]
    df = pd.DataFrame({"user_id": labels, "x": range(10)})
    with pytest.raises(TargetColumnError, match="explicit"):
        analyze_target_column(df, "user_id")


def test_analyze_high_cardinality_string_with_override_succeeds():
    labels = [f"user_{i}" for i in range(9)] + ["user_0"]
    df = pd.DataFrame({"user_id": labels, "x": range(10)})
    result = analyze_target_column(df, "user_id", override_task_type="classification")
    assert result.detected_task_type is None
    assert result.effective_task_type == "classification"
    assert result.overridden is True


def test_analyze_column_with_some_nulls_warns_but_succeeds():
    df = pd.DataFrame({"label": [0, 1, 0, 1, None, 1, 0], "x": range(7)})
    result = analyze_target_column(df, "label")
    assert result.effective_task_type == "classification"
    assert result.n_missing == 1
    assert any("missing value" in w for w in result.warnings)


def test_analyze_column_entirely_null_raises():
    df = pd.DataFrame({"label": [None, None, None], "x": range(3)})
    with pytest.raises(TargetColumnError, match="no non-null values"):
        analyze_target_column(df, "label")


def test_analyze_near_constant_column_raises_degenerate_error():
    df = pd.DataFrame({"label": [1, 1, 1, 1, 1, 1], "x": range(6)})
    with pytest.raises(TargetColumnError, match="degenerate"):
        analyze_target_column(df, "label")


def test_analyze_degenerate_after_dropping_nulls_raises():
    # Only one distinct non-null value remains once nulls are dropped.
    df = pd.DataFrame({"label": [1, None, 1, None, 1], "x": range(5)})
    with pytest.raises(TargetColumnError, match="degenerate"):
        analyze_target_column(df, "label")


def test_analyze_nonexistent_column_raises_hard_error_not_silent_fallback():
    """Regression test: an explicitly chosen target column that doesn't
    exist in the dataset must raise loudly, not silently fall back to
    guessing some other column (robustness rule: be strict with explicit,
    invalid user input)."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(TargetColumnError, match="not found"):
        analyze_target_column(df, "does_not_exist")


def test_analyze_unknown_override_task_type_raises():
    df = pd.DataFrame({"label": [0, 1, 0, 1], "x": range(4)})
    with pytest.raises(TargetColumnError, match="Unknown task_type"):
        analyze_target_column(df, "label", override_task_type="not-a-real-task-type")
