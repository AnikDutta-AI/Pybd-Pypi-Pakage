
"""Dataset visualization tools for the pybd package."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Support environments without a graphical display.
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Seaborn's default theme improves readability.
sns.set_theme(style="whitegrid", context="notebook")

_CATEGORICAL_DTYPES = ("object", "category", "string", "bool")

_SUPPORTED_FORMATS = {
    "png",
    "jpg",
    "jpeg",
    "svg",
    "pdf",
    "webp",
}


@dataclass(frozen=True)
class VisualizationConfig:
    """Configurable limits and rendering options for chart generation."""

    max_numeric_columns: int = 10
    max_categorical_columns: int = 10
    max_categories: int = 15
    histogram_bins: int | str = "auto"
    dpi: int = 120
    image_format: str = "png"
    annotate_heatmap_max_columns: int = 12

    def __post_init__(self) -> None:
        """Validate configuration values."""

        for name in (
            "max_numeric_columns",
            "max_categorical_columns",
            "max_categories",
            "dpi",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"'{name}' must be a positive integer."
                )

            if value < 1:
                raise ValueError(
                    f"'{name}' must be at least 1."
                )

        bins = self.histogram_bins

        if bins != "auto":
            if (
                isinstance(bins, bool)
                or not isinstance(bins, int)
                or bins < 1
            ):
                raise ValueError(
                    "histogram_bins must be 'auto' "
                    "or a positive integer."
                )

        image_format = self.image_format.lower().lstrip(".")

        if image_format not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {self.image_format!r}. "
                f"Supported formats: {sorted(_SUPPORTED_FORMATS)}."
            )

        object.__setattr__(self, "image_format", image_format)

        limit = self.annotate_heatmap_max_columns

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 0
        ):
            raise ValueError(
                "annotate_heatmap_max_columns "
                "must be a non-negative integer."
            )


def _safe_filename(name: object) -> str:
    """Convert a column name into a filesystem-safe filename."""

    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name))
    stem = stem.strip("_")

    return stem[:80] or "column"


class _ChartPathAllocator:
    """Allocate unique chart paths without filename collisions."""

    def __init__(
        self,
        directory: Path,
        image_format: str,
    ) -> None:
        self._directory = directory
        self._format = image_format.lstrip(".").lower()
        self._used: set[str] = set()

    def allocate(self, stem: str) -> Path:
        """Return a unique path for a chart."""

        candidate = stem
        suffix = 2

        while candidate.lower() in self._used:
            candidate = f"{stem}_{suffix}"
            suffix += 1

        self._used.add(candidate.lower())

        return self._directory / f"{candidate}.{self._format}"


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Remove missing and infinite values from numeric data."""

    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)

    return values.dropna()


def _plot_histogram(
    values: pd.Series,
    column: object,
    bins: int | str,
) -> Figure:
    """Create a Seaborn histogram with a density curve."""

    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()

    sns.histplot(
        data=values,
        bins=bins,
        kde=values.nunique() > 1 and len(values) > 1,
        color="steelblue",
        edgecolor="white",
        ax=ax,
    )

    ax.set_title(f"Distribution of {column}", fontsize=14)
    ax.set_xlabel(str(column))
    ax.set_ylabel("Frequency")

    fig.tight_layout()
    return fig


def _plot_boxplot(
    values: pd.Series,
    column: object,
) -> Figure:
    """Create a Seaborn boxplot for a numeric column."""

    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()

    sns.boxplot(
        x=values,
        color="lightblue",
        width=0.4,
        ax=ax,
    )

    ax.set_title(f"Box Plot of {column}", fontsize=14)
    ax.set_xlabel(str(column))

    fig.tight_layout()
    return fig


def _plot_missing_values(
    missing: pd.Series,
    total_rows: int,
) -> Figure:
    """Create a chart showing missing values and their percentages."""

    fig = Figure(figsize=(9, max(4, 0.4 * len(missing) + 2)))
    ax = fig.subplots()

    plot_data = pd.DataFrame({
        "Column": missing.index.astype(str),
        "Missing values": missing.to_numpy(),
    })

    sns.barplot(
        data=plot_data,
        x="Missing values",
        y="Column",
        color="coral",
        ax=ax,
    )

    labels = [
        f"{int(count)} ({count / total_rows:.1%})"
        for count in missing
    ]

    for patch, label in zip(ax.patches, labels):
        ax.annotate(
            label,
            (
                patch.get_width(),
                patch.get_y() + patch.get_height() / 2,
            ),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    ax.set_title("Missing Values by Column", fontsize=14)
    ax.set_xlabel("Number of missing values")
    ax.set_ylabel("Column")
    ax.margins(x=0.2)

    fig.tight_layout()
    return fig


def _plot_categories(
    counts: pd.Series,
    column: object,
    n_unique: int,
    max_categories: int,
) -> Figure:
    """Create a Seaborn horizontal bar chart of category counts."""

    ordered = counts.sort_values(ascending=True)

    fig = Figure(figsize=(9, max(4, 0.4 * len(ordered) + 2)))
    ax = fig.subplots()

    plot_data = pd.DataFrame({
        "Category": ordered.index.astype(str),
        "Count": ordered.to_numpy(),
    })

    sns.barplot(
        data=plot_data,
        x="Count",
        y="Category",
        color="cornflowerblue",
        ax=ax,
    )

    for patch in ax.patches:
        width = patch.get_width()
        ax.annotate(
            f"{int(width)}",
            (
                width,
                patch.get_y() + patch.get_height() / 2,
            ),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    if n_unique > max_categories:
        title = (
            f"Top {max_categories} of {n_unique} "
            f"Categories in {column}"
        )
    else:
        title = f"Categories in {column}"

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Count")
    ax.set_ylabel(str(column))
    ax.margins(x=0.12)

    fig.tight_layout()
    return fig


def _plot_correlation(
    correlation: pd.DataFrame,
    annotate_limit: int,
) -> Figure:
    """Create a Seaborn correlation heatmap."""

    size = len(correlation.columns)

    fig = Figure(
        figsize=(max(8, size), max(6, size))
    )
    ax = fig.subplots()

    sns.heatmap(
        correlation,
        annot=size <= annotate_limit,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )

    ax.set_title("Numeric Feature Correlation", fontsize=14)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    return fig


def create_visualizations(
    df: pd.DataFrame,
    output_dir: str | Path = "pybd_report/charts",
    config: VisualizationConfig | None = None,
) -> list[Path]:
    """Generate and save statistical charts for a DataFrame.

    Creates histograms, boxplots, missing-value charts,
    categorical value-count charts, and a correlation heatmap.

    Args:
        df: Dataset to visualize.
        output_dir: Directory where charts will be saved.
        config: Optional chart settings.

    Returns:
        Paths of generated chart files in creation order.

    Raises:
        TypeError: If df is not a DataFrame.
        ValueError: If df is empty or has duplicate column names.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )

    if df.empty:
        raise ValueError("Cannot visualize an empty dataset.")

    if not df.columns.is_unique:
        raise ValueError(
            "The dataset contains duplicate column names. "
            "Rename them before visualization."
        )

    if config is None:
        config = VisualizationConfig()
    elif not isinstance(config, VisualizationConfig):
        raise TypeError(
            "config must be a VisualizationConfig instance or None."
        )

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    allocator = _ChartPathAllocator(
        directory,
        config.image_format,
    )

    chart_files: list[Path] = []

    def save(fig: Figure, stem: str) -> None:
        """Save a chart, clear its figure, and record its path."""

        path = allocator.allocate(stem)

        try:
            fig.savefig(
                path,
                dpi=config.dpi,
                bbox_inches="tight",
                format=config.image_format,
            )

            chart_files.append(path)
            logger.debug("Saved chart: %s", path)

        finally:
            fig.clear()

    # Identify numeric and categorical columns.
    all_numeric = df.select_dtypes(
        include="number"
    ).columns.tolist()

    all_categorical = df.select_dtypes(
        include=list(_CATEGORICAL_DTYPES)
    ).columns.tolist()

    numeric_columns = all_numeric[:config.max_numeric_columns]

    categorical_columns = all_categorical[
        :config.max_categorical_columns
    ]

    if len(all_numeric) > len(numeric_columns):
        logger.info(
            "Visualizing the first %d of %d numeric columns.",
            len(numeric_columns),
            len(all_numeric),
        )

    if len(all_categorical) > len(categorical_columns):
        logger.info(
            "Visualizing the first %d of %d categorical columns.",
            len(categorical_columns),
            len(all_categorical),
        )

    # 1. Histograms and boxplots.
    for column in numeric_columns:
        values = _clean_numeric(df[column])

        if values.empty:
            logger.info(
                "Skipping numeric column %r: no finite values.",
                column,
            )
            continue

        stem = _safe_filename(column)

        save(
            _plot_histogram(
                values,
                column,
                config.histogram_bins,
            ),
            f"{stem}_histogram",
        )

        save(
            _plot_boxplot(values, column),
            f"{stem}_boxplot",
        )

    # 2. Missing-value overview.
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values()

    if not missing.empty:
        save(
            _plot_missing_values(missing, len(df)),
            "missing_values",
        )

    # 3. Categorical value counts.
    for column in categorical_columns:
        values = df[column].dropna()

        if values.empty:
            logger.info(
                "Skipping categorical column %r: no values.",
                column,
            )
            continue

        counts = (
            values.astype(str)
            .value_counts()
            .head(config.max_categories)
        )

        fig = _plot_categories(
            counts,
            column,
            int(values.nunique()),
            config.max_categories,
        )

        save(
            fig,
            f"{_safe_filename(column)}_categories",
        )

    # 4. Correlation heatmap.
    if numeric_columns:
        numeric_df = df[numeric_columns].copy()

        for column in numeric_columns:
            numeric_df[column] = _clean_numeric(
                numeric_df[column]
            )

        # Constant columns cannot provide meaningful correlations.
        numeric_df = numeric_df.loc[
            :,
            numeric_df.nunique(dropna=True) > 1,
        ]

        if numeric_df.shape[1] >= 2:
            correlation = numeric_df.corr()

            # Skip a heatmap when all correlations are undefined.
            if correlation.notna().any().any():
                save(
                    _plot_correlation(
                        correlation,
                        config.annotate_heatmap_max_columns,
                    ),
                    "correlation_heatmap",
                )

    logger.info(
        "Generated %d charts in %s.",
        len(chart_files),
        directory,
    )

    return chart_files
