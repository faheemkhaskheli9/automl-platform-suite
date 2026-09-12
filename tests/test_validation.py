import io

import pytest

from automl_core.validation import DatasetValidationError, validate_dataset_bytes


def test_valid_csv_returns_preview():
    content = b"a,b,c\n1,2,3\n4,5,6\n"
    preview = validate_dataset_bytes(content, "data.csv", preview_rows=10)

    assert preview.filename == "data.csv"
    assert preview.n_rows == 2
    assert preview.n_cols == 3
    assert preview.columns == ["a", "b", "c"]
    assert preview.dtypes["a"] == "int64"
    assert preview.preview_rows == [
        {"a": 1, "b": 2, "c": 3},
        {"a": 4, "b": 5, "c": 6},
    ]


def test_preview_is_capped_at_preview_rows():
    content = ("a\n" + "\n".join(str(i) for i in range(100))).encode()
    preview = validate_dataset_bytes(content, "data.csv", preview_rows=5)

    assert preview.n_rows == 100
    assert len(preview.preview_rows) == 5


def test_unsupported_extension_is_rejected():
    with pytest.raises(DatasetValidationError, match="Unsupported file type"):
        validate_dataset_bytes(b"whatever", "data.txt")


def test_empty_file_is_rejected():
    with pytest.raises(DatasetValidationError, match="empty"):
        validate_dataset_bytes(b"", "data.csv")


def test_malformed_csv_is_rejected_not_partially_imported():
    # Ragged rows: pandas' C parser raises rather than silently truncating.
    bad = b"a,b,c\n1,2\n3,4,5,6,7\n"
    with pytest.raises(DatasetValidationError):
        validate_dataset_bytes(bad, "data.csv")


def test_missing_header_is_rejected():
    content = b"a,,c\n1,2,3\n"
    with pytest.raises(DatasetValidationError, match="missing a header"):
        validate_dataset_bytes(content, "data.csv")


def test_duplicate_columns_are_rejected():
    content = b"a,a,b\n1,2,3\n"
    with pytest.raises(DatasetValidationError, match="Duplicate column"):
        validate_dataset_bytes(content, "data.csv")


def test_entirely_empty_column_is_rejected():
    content = b"a,b\n1,\n2,\n"
    with pytest.raises(DatasetValidationError, match="entirely empty"):
        validate_dataset_bytes(content, "data.csv")


def test_zero_row_file_is_rejected():
    content = b"a,b,c\n"
    with pytest.raises(DatasetValidationError, match="no data rows"):
        validate_dataset_bytes(content, "data.csv")


def test_xlsx_is_supported():
    pd = pytest.importorskip("pandas")
    openpyxl = pytest.importorskip("openpyxl")  # noqa: F841 - required engine

    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)

    preview = validate_dataset_bytes(buf.getvalue(), "data.xlsx")
    assert preview.columns == ["x", "y"]
    assert preview.n_rows == 2
