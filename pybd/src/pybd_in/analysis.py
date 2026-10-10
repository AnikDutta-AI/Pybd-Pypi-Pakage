"""Dataset analysis tools for the pybd package."""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["AnalysisConfig", "analyze_dataset"]

logger = logging.getLogger(__name__)

Report = dict[str, Any]

_CATEGORICAL_DTYPES = ["object", "category", "string", "boolean", "bool"]
_TEMPORAL_DTYPES = ["datetime", "datetimetz", "timedelta"]
_CORRELATION_METHODS = frozenset({"pearson", "spearman", "kendall"})


@dataclass(frozen=True)
class AnalysisConfig:
    """Tunable options for dataset analysis."""

    correlation_method: str = "pearson"
    strong_correlation_threshold: float = 0.9
    deep_memory_usage: bool = True
    round_digits: int = 2

    def __post_init__(self) -> None:
        if self.correlation_method not in _CORRELATION_METHODS:
            raise ValueError(
                f"'correlation_method' must be one of "
                f"{sorted(_CORRELATION_METHODS)}, got {self.correlation_method!r}."
            )
        if not 0 < self.strong_correlation_threshold <= 1:
            raise ValueError("'strong_correlation_threshold' must be in (0, 1].")
        if self.round_digits < 0:
            raise ValueError("'round_digits' must be non-negative.")


@dataclass(frozen=True)
class _ColumnGroups:
    numeric: list[Any]
    categorical: list[Any]
    temporal: list[Any]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _json_safe(value: Any) -> Any:
    """Convert NumPy, pandas, and stdlib values into JSON-serializable ones.

    Non-finite numbers and missing-value markers become ``None``; anything
    unrecognized falls back to its string representation.
    """
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe(item) for item in value), key=str)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.datetime64):
        return None if np.isnat(value) else str(value)
    # np.timedelta64 subclasses np.signedinteger, so test it before integers.
    if isinstance(value, np.timedelta64):
        return None if np.isnat(value) else str(pd.Timedelta(value))
    if isinstance(value, dt.timedelta):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating, Decimal)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return str(value)


def _validate_dataframe(df: object) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("Cannot analyze an empty dataset.")
    if not df.columns.is_unique:
        raise ValueError(
            "The dataset contains duplicate column names. Rename them before analysis."
        )


def _group_columns(df: pd.DataFrame) -> _ColumnGroups:
    return _ColumnGroups(
        numeric=df.select_dtypes(
            include="number", exclude="timedelta"
        ).columns.tolist(),
        categorical=df.select_dtypes(include=_CATEGORICAL_DTYPES).columns.tolist(),
        temporal=df.select_dtypes(include=_TEMPORAL_DTYPES).columns.tolist(),
    )


def _percentage(part: float, whole: float, digits: int) -> float:
    return round(float(part) / float(whole) * 100, digits)


def _stringify(series: pd.Series) -> pd.Series:
    """Stringify values (for unhashable cells such as lists), keeping NaN."""
    return series.astype(str).where(series.notna())


def _finite_numeric(df: pd.DataFrame, columns: list[Any]) -> pd.DataFrame:
    """Return the given columns as float64 with +/-inf replaced by NaN."""
    return df[columns].astype("float64").replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Report sections
# --------------------------------------------------------------------------- #
def _missing_values(df: pd.DataFrame, config: AnalysisConfig) -> dict[str, Any]:
    counts = df.isna().sum()
    return {
        str(column): {
            "count": int(count),
            "percentage": _percentage(count, len(df), config.round_digits),
        }
        for column, count in counts.items()
    }


def _count_duplicates(df: pd.DataFrame) -> int:
    try:
        return int(df.duplicated().sum())
    except TypeError:
        logger.warning(
            "Unhashable values found; comparing rows by string representation."
        )
        comparable = df.apply(
            lambda col: _stringify(col) if col.dtype == object else col
        )
        return int(comparable.duplicated().sum())


def _count_unique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(_stringify(series).nunique(dropna=True))


def _count_infinite(df: pd.DataFrame, numeric_columns: list[Any]) -> int:
    if not numeric_columns:
        return 0
    return int(np.isinf(df[numeric_columns].astype("float64")).sum().sum())


def _numeric_statistics(
    df: pd.DataFrame, columns: list[Any]
) -> dict[str, dict[str, Any]]:
    """Summary statistics over finite values, plus skewness and inf counts."""
    if not columns:
        return {}

    numeric = df[columns].astype("float64")
    infinite = np.isinf(numeric).sum()
    finite = numeric.replace([np.inf, -np.inf], np.nan)
    summary = finite.describe().transpose()
    skewness = finite.skew()

    statistics: dict[str, dict[str, Any]] = {}
    for column in columns:
        row = {
            str(stat): _json_safe(value)
            for stat, value in summary.loc[column].items()
        }
        row["skewness"] = _json_safe(skewness[column])
        row["infinite_count"] = int(infinite[column])
        statistics[str(column)] = row
    return statistics


def _categorical_statistics(
    df: pd.DataFrame, columns: list[Any]
) -> dict[str, dict[str, Any]]:
    statistics: dict[str, dict[str, Any]] = {}
    for column in columns:
        series = df[column]
        try:
            described = series.describe()
        except TypeError:
            described = _stringify(series).describe()
        statistics[str(column)] = {
            str(stat): _json_safe(value) for stat, value in described.items()
        }
    return statistics


def _temporal_statistics(
    df: pd.DataFrame, columns: list[Any]
) -> dict[str, dict[str, Any]]:
    return {
        str(column): {
            "count": int(df[column].count()),
            "min": _json_safe(df[column].min()),
            "max": _json_safe(df[column].max()),
        }
        for column in columns
    }


def _correlation(
    df: pd.DataFrame, numeric_columns: list[Any], config: AnalysisConfig
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Return the correlation matrix and the strongly correlated column pairs."""
    if len(numeric_columns) < 2:
        return {}, []

    numeric = _finite_numeric(df, numeric_columns)
    # Constant columns have undefined correlation.
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if numeric.shape[1] < 2:
        return {}, []

    correlation = numeric.corr(method=config.correlation_method)
    matrix = {
        str(column): {str(other): _json_safe(value) for other, value in row.items()}
        for column, row in correlation.iterrows()
    }

    columns = list(correlation.columns)
    pairs = [
        {
            "column_a": str(a),
            "column_b": str(b),
            "correlation": round(float(correlation.at[a, b]), 4),
        }
        for i, a in enumerate(columns)
        for b in columns[i + 1 :]
        if pd.notna(correlation.at[a, b])
        and abs(correlation.at[a, b]) >= config.strong_correlation_threshold
    ]
    pairs.sort(key=lambda pair: abs(pair["correlation"]), reverse=True)
    return matrix, pairs


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def analyze_dataset(
    df: pd.DataFrame, config: AnalysisConfig | None = None
) -> Report:
    """Analyze a DataFrame and return a JSON-serializable dataset report.

    Args:
        df: The pandas DataFrame to analyze.
        config: Optional analysis options; defaults are used if omitted.

    Returns:
        A dictionary with the sections ``dataset_overview``, ``data_quality``,
        ``column_details``, ``descriptive_statistics`` (only statistics that
        apply to each column's type), ``correlation_matrix``, and
        ``strong_correlations``.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If ``df`` is empty or has duplicate column names.
    """
    _validate_dataframe(df)
    config = config or AnalysisConfig()

    row_count, column_count = df.shape
    logger.info("Analyzing dataset with %d rows and %d columns.", row_count, column_count)

    groups = _group_columns(df)
    digits = config.round_digits

    total_cells = row_count * column_count
    total_missing = int(df.isna().sum().sum())
    duplicate_count = _count_duplicates(df)
    complete_rows = int((~df.isna().any(axis=1)).sum())

    statistics = {
        **_numeric_statistics(df, groups.numeric),
        **_categorical_statistics(df, groups.categorical),
        **_temporal_statistics(df, groups.temporal),
    }
    # Preserve the original column order.
    statistics = {
        str(column): statistics[str(column)]
        for column in df.columns
        if str(column) in statistics
    }

    correlation_matrix, strong_correlations = _correlation(df, groups.numeric, config)

    report: Report = {
        "dataset_overview": {
            "rows": int(row_count),
            "columns": int(column_count),
            "total_cells": int(total_cells),
            "memory_usage_bytes": int(
                df.memory_usage(index=True, deep=config.deep_memory_usage).sum()
            ),
        },
        "data_quality": {
            "missing_cells": total_missing,
            "missing_percentage": _percentage(total_missing, total_cells, digits),
            "duplicate_rows": duplicate_count,
            "duplicate_percentage": _percentage(duplicate_count, row_count, digits),
            "complete_rows": complete_rows,
            "infinite_cells": _count_infinite(df, groups.numeric),
        },
        "column_details": {
            "data_types": {str(c): str(t) for c, t in df.dtypes.items()},
            "unique_counts": {str(c): _count_unique(df[c]) for c in df.columns},
            "missing_values": _missing_values(df, config),
            "numeric_columns": [str(c) for c in groups.numeric],
            "categorical_columns": [str(c) for c in groups.categorical],
            "temporal_columns": [str(c) for c in groups.temporal],
        },
        "descriptive_statistics": statistics,
        "correlation_matrix": correlation_matrix,
        "strong_correlations": strong_correlations,
    }

    logger.info("Dataset analysis completed successfully.")
    return report