"""Dataset profiling for the pybd package."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from pybd_in.analysis import AnalysisConfig, analyze_dataset
from pybd_in.dataset import load_dataset
from pybd_in.preprocessing import (
    PreprocessingConfig,
    Recommendation,
    Severity,
    recommend_preprocessing,
)

if TYPE_CHECKING:
    # Imported lazily at runtime so profiles without charts never load matplotlib.
    from pybd_in.visualization import VisualizationConfig

__all__ = ["DatasetProfile", "profile_dataset"]

logger = logging.getLogger(__name__)

_NUMERIC_STAT_COLUMNS = ("count", "mean", "std", "min", "50%", "max")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _format_bytes(size: float) -> str:
    if size < 1024:
        return f"{size:,.0f} B"
    for unit in ("KB", "MB", "GB"):
        size /= 1024
        if size < 1024:
            return f"{size:,.2f} {unit}"
    return f"{size / 1024:,.2f} TB"


def _heading(title: str) -> str:
    return f"=== {title.upper()} ==="


def _relative(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _generate_charts(
    df: pd.DataFrame,
    charts_dir: Path,
    config: VisualizationConfig | None,
) -> tuple[tuple[Path, ...], str | None]:
    """Create charts, returning ``(paths, skipped_reason)``.

    Chart dependencies are optional: if they cannot be imported, charts are
    skipped with a warning instead of failing the whole profile.
    """
    try:
        from pybd_in.visualization import create_visualizations
    except ImportError as exc:
        logger.warning("Skipping charts: visualization dependencies unavailable (%s).", exc)
        return (), str(exc)

    return tuple(create_visualizations(df, output_dir=charts_dir, config=config)), None


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class DatasetProfile:
    """Immutable result of profiling a dataset.

    Build one with :func:`profile_dataset` or :meth:`DatasetProfile.from_file`.
    ``data`` is held by reference, not copied, so avoid mutating the
    DataFrame after profiling if you rely on the profile staying consistent.
    """

    source: Path
    output_dir: Path
    data: pd.DataFrame
    analysis: dict[str, Any]
    recommendations: tuple[Recommendation, ...]
    charts: tuple[Path, ...] = ()
    charts_skipped_reason: str | None = None

    # ----- construction --------------------------------------------------- #
    @classmethod
    def from_file(
        cls,
        file_path: str | Path,
        output_dir: str | Path = "pybd_report",
        generate_charts: bool = True,
        *,
        analysis_config: AnalysisConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        visualization_config: VisualizationConfig | None = None,
    ) -> DatasetProfile:
        """Load a dataset from disk and profile it.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
            ValueError: If the dataset cannot be loaded or analyzed.
            OSError: If chart files cannot be written.
        """
        source = Path(file_path)
        output = Path(output_dir)

        if not source.is_file():
            raise FileNotFoundError(f"Input file not found: {source}")

        logger.info("Profiling %s.", source)
        data = load_dataset(source)
        analysis = analyze_dataset(data, analysis_config)
        recommendations = tuple(recommend_preprocessing(data, preprocessing_config))

        charts: tuple[Path, ...] = ()
        skipped_reason: str | None = None
        if generate_charts:
            charts, skipped_reason = _generate_charts(
                data, output / "charts", visualization_config
            )

        return cls(
            source=source,
            output_dir=output,
            data=data,
            analysis=analysis,
            recommendations=recommendations,
            charts=charts,
            charts_skipped_reason=skipped_reason,
        )

    # ----- properties ----------------------------------------------------- #
    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, columns)`` of the profiled dataset."""
        return self.data.shape

    @property
    def has_critical_issues(self) -> bool:
        """True if any recommendation has CRITICAL severity."""
        return any(item.severity is Severity.CRITICAL for item in self.recommendations)

    def column_summary(self) -> pd.DataFrame:
        """One row per column: dtype, missing count/percentage, unique values."""
        details = self.analysis["column_details"]
        rows = {
            column: {
                "dtype": dtype,
                "missing": details["missing_values"][column]["count"],
                "missing_%": details["missing_values"][column]["percentage"],
                "unique": details["unique_counts"][column],
            }
            for column, dtype in details["data_types"].items()
        }
        summary = pd.DataFrame.from_dict(rows, orient="index")
        summary.index.name = "column"
        return summary

    # ----- text rendering ------------------------------------------------- #
    def render_overview(self, preview_rows: int = 5) -> str:
        """Dataset shape, memory use, per-column summary, and a data preview."""
        overview = self.analysis["dataset_overview"]
        lines = [
            _heading("Dataset overview"),
            f"File:          {self.source.name}",
            f"Rows:          {overview['rows']:,}",
            f"Columns:       {overview['columns']:,}",
            f"Memory usage:  {_format_bytes(overview['memory_usage_bytes'])}",
            "",
            "Columns:",
            self.column_summary().to_string(),
        ]
        if preview_rows > 0:
            lines += [
                "",
                f"First {preview_rows} rows:",
                self.data.head(preview_rows).to_string(max_cols=20, max_colwidth=40),
            ]
        return "\n".join(lines)

    def render_analysis(self) -> str:
        """Data-quality indicators, numeric statistics, and strong correlations."""
        quality = self.analysis["data_quality"]
        lines = [
            _heading("Analysis report"),
            f"Missing cells:   {quality['missing_cells']:,} ({quality['missing_percentage']}%)",
            f"Duplicate rows:  {quality['duplicate_rows']:,} ({quality['duplicate_percentage']}%)",
            f"Complete rows:   {quality['complete_rows']:,}",
        ]
        if quality.get("infinite_cells"):
            lines.append(f"Infinite cells:  {quality['infinite_cells']:,}")

        numeric = set(self.analysis["column_details"]["numeric_columns"])
        stats = {
            column: {name: values.get(name) for name in _NUMERIC_STAT_COLUMNS}
            for column, values in self.analysis["descriptive_statistics"].items()
            if column in numeric
        }
        if stats:
            table = pd.DataFrame.from_dict(stats, orient="index")
            lines += [
                "",
                "Numeric statistics:",
                table.to_string(float_format=lambda value: f"{value:,.3f}"),
            ]

        lines += ["", "Strongly correlated pairs:"]
        pairs = self.analysis.get("strong_correlations", [])
        if pairs:
            lines += [
                f"  {pair['column_a']} <-> {pair['column_b']}: {pair['correlation']:+.2f}"
                for pair in pairs
            ]
        else:
            lines.append("  none above the configured threshold")
        return "\n".join(lines)

    def render_recommendations(self) -> str:
        """Numbered preprocessing recommendations, most severe first."""
        lines = [_heading("Recommendations")]
        if not self.recommendations:
            lines.append("No recommendations were generated.")
        else:
            lines += [
                f"{number}. {item}"
                for number, item in enumerate(self.recommendations, start=1)
            ]
        return "\n".join(lines)

    def render_visualizations(self) -> str:
        """Paths of the generated charts, or why none exist."""
        lines = [_heading("Visualizations")]
        if self.charts:
            lines += [str(path) for path in self.charts]
        elif self.charts_skipped_reason:
            lines.append(f"Charts skipped: {self.charts_skipped_reason}")
        else:
            lines.append("No charts were generated.")
        return "\n".join(lines)

    def render_report(self, preview_rows: int = 5) -> str:
        """All sections as one string, suitable for logging or saving."""
        return "\n\n".join(
            (
                self.render_overview(preview_rows),
                self.render_analysis(),
                self.render_recommendations(),
                self.render_visualizations(),
            )
        )

    # ----- display (thin wrappers that print the rendered text) ----------- #
    def show_overview(self, preview_rows: int = 5) -> None:
        print(self.render_overview(preview_rows))

    def show_analysis(self) -> None:
        print(self.render_analysis())

    def show_recommendations(self) -> None:
        print(self.render_recommendations())

    def show_visualizations(self) -> None:
        print(self.render_visualizations())

    def show_report(self, preview_rows: int = 5) -> None:
        print(self.render_report(preview_rows))

    # ----- serialization -------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary (the raw DataFrame is not included)."""
        return {
            "source": str(self.source),
            "analysis": self.analysis,
            "recommendations": [
                {
                    "category": item.category,
                    "severity": item.severity.name,
                    "message": item.message,
                    "columns": list(item.columns),
                }
                for item in self.recommendations
            ],
            "charts": [_relative(path, self.output_dir) for path in self.charts],
        }

    def to_json(self, indent: int | None = 4) -> str:
        """Serialize :meth:`to_dict` to a strict JSON string."""
        return json.dumps(
            self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False
        )

    def __repr__(self) -> str:
        rows, columns = self.shape
        return (
            f"DatasetProfile(source={str(self.source)!r}, rows={rows}, "
            f"columns={columns}, recommendations={len(self.recommendations)}, "
            f"charts={len(self.charts)})"
        )


def profile_dataset(
    file_path: str | Path,
    output_dir: str | Path = "pybd_report",
    generate_charts: bool = True,
    *,
    analysis_config: AnalysisConfig | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
    visualization_config: VisualizationConfig | None = None,
) -> DatasetProfile:
    """Create and return a complete dataset profile.

    Args:
        file_path: Path to the input dataset.
        output_dir: Directory where charts are saved (under ``charts/``).
        generate_charts: Whether to generate visualization charts.
        analysis_config: Optional analysis settings.
        preprocessing_config: Optional preprocessing-check thresholds.
        visualization_config: Optional chart settings.

    Returns:
        A :class:`DatasetProfile`.
    """
    return DatasetProfile.from_file(
        file_path,
        output_dir,
        generate_charts,
        analysis_config=analysis_config,
        preprocessing_config=preprocessing_config,
        visualization_config=visualization_config,
    )