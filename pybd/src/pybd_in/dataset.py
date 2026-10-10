
from pathlib import Path
import stat

import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a CSV dataset after validating the file."""

    path = Path(file_path)

    try:
        file_stat = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        ) from exc
    except (NotADirectoryError, PermissionError, OSError) as exc:
        raise ValueError(
            f"Unable to access dataset path: {path}. {exc}"
        ) from exc

    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(
            f"The provided path is not a regular file: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            "Unsupported dataset format. Please provide a CSV file."
        )

    if file_stat.st_size == 0:
        raise ValueError("The dataset file is empty.")

    try:
        df = pd.read_csv(
            path,
            engine="c",
            low_memory=True,
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            "The dataset file is empty or contains no readable data."
        ) from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            "Unable to parse the CSV file. Please check its structure."
        ) from exc
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Unable to decode the CSV file. Check its text encoding."
        ) from exc

    if df.shape[1] == 0:
        raise ValueError("The dataset contains no columns.")

    if df.empty:
        raise ValueError("The dataset contains no data rows.")

    return df
