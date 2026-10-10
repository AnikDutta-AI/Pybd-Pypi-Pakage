
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from pybd_in.analysis import analyze_dataset
from pybd_in.dataset import load_dataset
from pybd_in.preprocessing import recommend_preprocessing
from pybd_in.visualization import create_visualizations


logger = logging.getLogger(__name__)


def run_pipeline(
    file_path: str | Path,
    output_dir: str | Path = "pybd_report",
    generate_charts: bool = True,
) -> dict[str, Any]:
    """
    Run the complete pybd data analysis pipeline.

    Steps:
        1. Load the CSV dataset.
        2. Analyze dataset quality and statistics.
        3. Generate preprocessing recommendations.
        4. Create visualizations if enabled.
        5. Save the final report as JSON.

    Args:
        file_path: Path to the input CSV file.
        output_dir: Directory where reports and charts are saved.
        generate_charts: Whether to generate visualization charts.

    Returns:
        A dictionary containing the analysis report and output paths.
    """
    input_path = Path(file_path)
    report_dir = Path(output_dir)

    logger.info("Starting pybd analysis pipeline.")

    # Step 1: Load the dataset
    logger.info("Loading dataset: %s", input_path)
    df = load_dataset(input_path)

    # Step 2: Analyze the dataset
    logger.info("Analyzing dataset.")
    analysis_report = analyze_dataset(df)

    # Step 3: Generate preprocessing recommendations
    logger.info("Generating preprocessing recommendations.")
    recommendations = recommend_preprocessing(df)

    recommendation_report = [
        {
            "category": item.category,
            "severity": item.severity.name,
            "message": item.message,
            "columns": list(item.columns),
        }
        for item in recommendations
    ]

    # Step 4: Generate visualizations
    chart_paths: list[str] = []

    if generate_charts:
        logger.info("Generating visualizations.")

        generated_charts = create_visualizations(
            df,
            output_dir=report_dir / "charts",
        )

        chart_paths = [
            str(path)
            for path in generated_charts
        ]

        logger.info("Generated %d charts.", len(chart_paths))
    else:
        logger.info("Chart generation skipped.")

    # Step 5: Build the final report
    report: dict[str, Any] = {
        "package": "pybd",
        "input_file": str(input_path),
        "analysis": analysis_report,
        "preprocessing_recommendations": recommendation_report,
        "visualizations": {
            "generated": generate_charts,
            "chart_count": len(chart_paths),
            "chart_paths": chart_paths,
        },
    }

    # Step 6: Save the report as JSON
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = report_dir / "report.json"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    logger.info("Report saved to: %s", report_path)

    # Include the report location in the returned result
    report["report_path"] = str(report_path)

    logger.info("pybd analysis pipeline completed successfully.")

    return report


def main() -> None:
    """Run the pybd command-line application."""

    parser = argparse.ArgumentParser(
        prog="pybd",
        description=(
            "Analyze a CSV dataset, generate statistics, "
            "recommend preprocessing, and create charts."
        ),
    )

    parser.add_argument(
        "file",
        help="Path to the input CSV dataset.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="pybd_report",
        help="Directory for the JSON report and charts.",
    )

    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip visualization generation.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        report = run_pipeline(
            file_path=args.file,
            output_dir=args.output,
            generate_charts=not args.no_charts,
        )

        print("\nAnalysis completed successfully.")
        print(f"Report saved to: {report['report_path']}")

        print(
            "Charts generated: "
            f"{report['visualizations']['chart_count']}"
        )

        print(
            "Preprocessing recommendations: "
            f"{len(report['preprocessing_recommendations'])}"
        )

    except (ValueError, TypeError, OSError) as exc:
        logger.error("Analysis failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
