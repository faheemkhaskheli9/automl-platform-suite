"""
Dataset upload validation shared by every AutoML feature (train/tune/evaluate).

Per README.md Section 2, `automl_core` is the one upload path all 3 feature
apps sit on top of, so this module has no Django/view dependencies — it is
pure pandas + stdlib and is reused as-is by later phases.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import BinaryIO

import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")


class DatasetValidationError(ValueError):
    """Raised when an uploaded file fails validation.

    Carries a short, user-facing `reason` distinct from `str(exc)` so a view
    can show a clear message without leaking a raw parser traceback.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass
class DatasetPreview:
    """Result of a successful validation: enough to render a preview table."""

    filename: str
    n_rows: int
    n_cols: int
    columns: list[str]
    dtypes: dict[str, str]
    preview_rows: list[dict] = field(default_factory=list)


def _extension(filename: str) -> str:
    idx = filename.rfind(".")
    return filename[idx:].lower() if idx != -1 else ""


def _raw_header(fileobj: BinaryIO, ext: str) -> list[str]:
    """Read just the header row as written in the file, before pandas gets a
    chance to mangle duplicate/blank names (`a, a` -> `a, a.1`) — duplicate
    and missing-header checks need the *original* names, not the mangled
    ones, or every file with dupes/blanks would silently pass."""
    pos = fileobj.tell()
    try:
        if ext == ".csv":
            first_line = fileobj.readline()
            if isinstance(first_line, bytes):
                first_line = first_line.decode("utf-8-sig", errors="replace")
            return next(csv.reader(io.StringIO(first_line)), [])
        else:
            import openpyxl

            wb = openpyxl.load_workbook(fileobj, read_only=True)
            try:
                sheet = wb.active
                row = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True), ())
                return ["" if v is None else str(v) for v in row]
            finally:
                wb.close()
    finally:
        fileobj.seek(pos)


def _read_dataframe(fileobj: BinaryIO, filename: str) -> pd.DataFrame:
    ext = _extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise DatasetValidationError(
            f"Unsupported file type '{ext or '(none)'}'. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    raw_columns = _raw_header(fileobj, ext)
    if any(not c.strip() for c in raw_columns):
        raise DatasetValidationError("One or more columns are missing a header name.")
    if len(set(raw_columns)) != len(raw_columns):
        dupes = sorted({c for c in raw_columns if raw_columns.count(c) > 1})
        raise DatasetValidationError(f"Duplicate column name(s): {', '.join(dupes)}.")

    try:
        if ext == ".csv":
            # Read the whole file up front (no chunked/streaming parse) so a
            # mid-file parse error surfaces here rather than after a caller
            # has already accepted a partially-read result.
            df = pd.read_csv(fileobj)
        else:
            df = pd.read_excel(fileobj)
    except pd.errors.EmptyDataError as exc:
        raise DatasetValidationError("The file is empty.") from exc
    except (pd.errors.ParserError, ValueError, UnicodeDecodeError) as exc:
        raise DatasetValidationError(f"Could not parse file as {ext}: {exc}") from exc

    return df


def validate_dataset(fileobj: BinaryIO, filename: str, *, preview_rows: int = 10) -> DatasetPreview:
    """Validate an uploaded tabular file and return a preview.

    Never returns a partial result: the dataframe is fully parsed and every
    check below passes, or a `DatasetValidationError` is raised and nothing
    is returned/cached.
    """
    df = _read_dataframe(fileobj, filename)

    if df.shape[1] == 0:
        raise DatasetValidationError("The file has no columns.")
    if df.shape[0] == 0:
        raise DatasetValidationError("The file has no data rows.")

    # Header shape (missing/duplicate names) was already checked in
    # _read_dataframe against the raw header, before pandas could mangle it.
    columns = [str(c) for c in df.columns]

    empty_cols = [c for c in columns if df[c].isna().all()]
    if empty_cols:
        raise DatasetValidationError(
            f"Column(s) entirely empty: {', '.join(empty_cols)}."
        )

    dtypes = {c: str(df[c].dtype) for c in columns}
    head = df.head(preview_rows)
    # NaN -> None so the preview serializes cleanly (json/template) without
    # producing the string "nan".
    preview = head.where(pd.notnull(head), None).to_dict(orient="records")

    return DatasetPreview(
        filename=filename,
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        columns=columns,
        dtypes=dtypes,
        preview_rows=preview,
    )


def validate_dataset_bytes(content: bytes, filename: str, *, preview_rows: int = 10) -> DatasetPreview:
    """Convenience wrapper for callers holding raw bytes (e.g. Django's
    `UploadedFile.read()`) instead of a seekable file object."""
    return validate_dataset(io.BytesIO(content), filename, preview_rows=preview_rows)
