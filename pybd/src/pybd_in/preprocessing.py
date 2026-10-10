
"""Dataset quality checks that produce actionable preprocessing recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

import pandas as pd


class Severity(IntEnum):
    """Ordered so that higher values indicate greater urgency."""

    INFO = 1
    WARNING = 2
    CRITICAL = 3


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configurable thresholds for dataset quality checks."""

    high_cardinality_threshold: int = 50
    critical_missing_pct: float = 50.0
    scale_ratio_threshold: float = 10.0

    def __post_init__(self) -> None:
        if self.high_cardinality_threshold < 1:
            raise ValueError(
                "high_cardinality_threshold must be at least 1."
            )

        if not 0 <= self.critical_missing_pct <= 100:
            raise ValueError(
                "critical_missing_pct must be between 0 and 100."
            )

        if self.scale_ratio_threshold <= 1:
            raise ValueError(
                "scale_ratio_threshold must be greater than 1."
            )


@dataclass(frozen=True)
class Recommendation:
    """A single preprocessing recommendation."""

    category: str
    severity: Severity
    message: str
    columns: tuple[str, ...] = ()

    def __str__(self) -> str:
        return (
            f"[{self.severity.name}] "
            f"{self.category}: {self.message}"
        )


Check = Callable[
    [pd.DataFrame, PreprocessingConfig],
    list[Recommendation],
]


def _check_missing_values(
    df: pd.DataFrame,
    config: PreprocessingConfig,
) -> list[Recommendation]:
    """Identify columns containing missing values."""

    results: list[Recommendation] = []
    missing = df.isna().sum()

    for column, count in missing[missing > 0].items():
        percentage = float(count / len(df) * 100)

        if pd.api.types.is_numeric_dtype(df[column]):
            method = "median or mean imputation"
        else:
            method = "mode imputation or an 'Unknown' category"

        severity = (
            Severity.CRITICAL
            if percentage >= config.critical_missing_pct
            else Severity.WARNING
        )

        advice = f"Consider {method}."

        if severity == Severity.CRITICAL:
            advice += (
                " Consider dropping the column if it is not essential."
            )

        results.append(
            Recommendation(
                category="Missing values",
                severity=severity,
                message=(
                    f"'{column}' has {count} missing values "
                    f"({percentage:.2f}%). {advice}"
                ),
                columns=(str(column),),
            )
        )

    return results


def _check_duplicates(
    df: pd.DataFrame,
    config: PreprocessingConfig,
) -> list[Recommendation]:
    """Identify duplicate rows."""

    count = int(df.duplicated().sum())

    if count == 0:
        return []

    return [
        Recommendation(
            category="Duplicates",
            severity=Severity.WARNING,
            message=(
                f"Found {count} duplicate rows. Review them before "
                "deciding whether to remove them."
            ),
        )
    ]


def _check_categorical_encoding(
    df: pd.DataFrame,
    config: PreprocessingConfig,
) -> list[Recommendation]:
    """Identify categorical and Boolean columns."""

    columns = df.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns

    if columns.empty:
        return []

    column_names = [str(column) for column in columns]

    return [
        Recommendation(
            category="Encoding",
            severity=Severity.INFO,
            message=(
                "Review categorical columns before modeling. "
                f"Consider appropriate encoding for: {column_names}."
            ),
            columns=tuple(column_names),
        )
    ]


def _check_high_cardinality(
    df: pd.DataFrame,
    config: PreprocessingConfig,
) -> list[Recommendation]:
    """Identify categorical columns with many unique values."""

    categorical = df.select_dtypes(
        include=["object", "category", "string", "bool"]
    )

    columns = [
        str(column)
        for column in categorical.columns
        if categorical[column].nunique(dropna=True)
        > config.high_cardinality_threshold
    ]

    if not columns:
        return []

    return [
        Recommendation(
            category="High cardinality",
            severity=Severity.WARNING,
            message=(
                f"Columns {columns} exceed "
                f"{config.high_cardinality_threshold} unique values. "
                "One-hot encoding may create too many features. "
                "Consider frequency encoding, hashing, or another "
                "method suitable for your model and data."
            ),
            columns=tuple(columns),
        )
    ]


def _check_feature_scaling(
    df: pd.DataFrame,
    config: PreprocessingConfig,
) -> list[Recommendation]:
    """Check whether numeric features have substantially different spreads."""

    numeric = df.select_dtypes(include="number")

    if numeric.shape[1] < 2:
        return []

    # Ignore missing standard deviations and constant columns.
    spread = numeric.std().dropna()
    spread = spread[spread > 0]

    if len(spread) < 2:
        return []

    ratio = float(spread.max() / spread.min())

    if ratio < config.scale_ratio_threshold:
        return []

    return [
        Recommendation(
            category="Scaling",
            severity=Severity.INFO,
            message=(
                f"Numeric feature standard deviations differ by "
                f"approximately {ratio:,.1f}x. Consider StandardScaler "
                "or MinMaxScaler for scale-sensitive algorithms such "
                "as SVM, KNN, and logistic regression. This is a "
                "heuristic, not a requirement."
            ),
            columns=tuple(str(column) for column in spread.index),
        )
    ]


def _check_constant_columns(
    df: pd.DataFrame,
    config: PreprocessingConfig,
) -> list[Recommendation]:
    """Identify columns containing only one distinct value."""

    columns = [
        str(column)
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    if not columns:
        return []

    return [
        Recommendation(
            category="Constant columns",
            severity=Severity.WARNING,
            message=(
                f"Columns {columns} contain a single distinct value. "
                "Consider removing them if they provide no useful "
                "information for your task."
            ),
            columns=tuple(columns),
        )
    ]


_CHECKS: tuple[Check, ...] = (
    _check_missing_values,
    _check_duplicates,
    _check_constant_columns,
    _check_categorical_encoding,
    _check_high_cardinality,
    _check_feature_scaling,
)


def recommend_preprocessing(
    df: pd.DataFrame,
    config: PreprocessingConfig | None = None,
) -> list[Recommendation]:
    """Analyze a DataFrame and return preprocessing recommendations.

    Args:
        df: The dataset to analyze.
        config: Optional thresholds for the quality checks.

    Returns:
        Recommendations sorted from highest to lowest severity.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If df is empty or contains duplicate column names.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )

    if df.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    if not df.columns.is_unique:
        raise ValueError(
            "The dataset contains duplicate column names. "
            "Rename duplicate columns before analysis."
        )

    if config is None:
        config = PreprocessingConfig()
    elif not isinstance(config, PreprocessingConfig):
        raise TypeError(
            "config must be a PreprocessingConfig instance or None."
        )

    recommendations = [
        recommendation
        for check in _CHECKS
        for recommendation in check(df, config)
    ]

    recommendations.sort(
        key=lambda recommendation: recommendation.severity,
        reverse=True,
    )

    if not recommendations:
        recommendations.append(
            Recommendation(
                category="General",
                severity=Severity.INFO,
                message=(
                    "No obvious preprocessing issues were detected. "
                    "Validate the data before training a model."
                ),
            )
        )

    return recommendations


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "age": [25, None, 40],
            "income": [50_000, 60_000, 1_000_000],
            "city": ["A", "B", "A"],
        }
    )

    for recommendation in recommend_preprocessing(sample):
        print(recommendation)
