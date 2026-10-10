# pybd

**Dataset Analysis, Data Quality Assessment, and Preprocessing Recommendations for Python**

`pybd-in` is a Python package designed to help data science beginners and developers explore datasets, analyze data quality, identify potential issues, and get preprocessing recommendations with simple Python functions.

Instead of writing repetitive code for basic dataset inspection, you can use `pybd-in` to get started with your data analysis workflow more easily.

## Features

- **Dataset Loading:** Load CSV datasets into pandas DataFrames.
- **Dataset Analysis:** Inspect datasets and obtain analysis results.
- **Data Quality Assessment:** Profile datasets to understand their structure and quality.
- **Preprocessing Recommendations:** Get suggestions for handling common data preparation issues.
- **Dataset Profiling:** Generate a structured profile using `DatasetProfile` and `profile_dataset`.
- **Data Visualization:** Generate dataset charts, including distributions, boxplots, categorical counts, missing-value charts when applicable, and correlation heatmaps.
- **Pandas Compatibility:** Work with the returned pandas DataFrame using familiar methods such as `.head()`, `.shape`, and `.info()`.

## Installation

Install the package from PyPI:

```bash
python -m pip install pybd-in
```

For visualization features, install the optional dependencies:

```bash
python -m pip install "pybd-in[visualization]"
```

For development and testing:

```bash
python -m pip install "pybd-in[dev]"
```

## Quick Start

### 1. Import the package

```python
import pybd_in as bd
```

### 2. Load a dataset

```python
df = bd.load_dataset("sample_dataset.csv")

print(df.head())
print("Dataset shape:", df.shape)
```

Replace `sample_dataset.csv` with the path to your own CSV file.

### 3. Analyze the dataset

```python
analysis = bd.analyze_dataset(df)

print(analysis)
```

This function returns dataset analysis results. The exact information depends on the implementation and the input dataset.

### 4. Get preprocessing recommendations

```python
recommendations = bd.recommend_preprocessing(df)

for recommendation in recommendations:
    print(recommendation)
```

Use the recommendations to help decide which preprocessing steps may be appropriate for your data.

### 5. Profile the dataset

```python
profile = bd.profile_dataset(df)

print(profile)
```

You can also import the profiling class directly:

```python
from pybd_in import DatasetProfile, profile_dataset
```

### 6. Generate visualizations

First install the optional visualization dependencies:

```bash
python -m pip install "pybd-in[visualization]"
```

Then generate charts:

```python
charts = bd.create_visualizations(df)

for chart in charts:
    print(chart)
```

The function returns the paths to the generated chart files. The default output directory is `pybd_report/charts`.

## Example Dataset

```python
import pandas as pd
import pybd_in as bd

df = pd.DataFrame({
    "Age": [20, 21, 22, 23, 24, 25],
    "Salary": [25000, 30000, None, 40000, 45000, 50000],
    "Department": ["IT", "HR", "IT", "Sales", "HR", "IT"],
    "Score": [75, 80, 85, 90, 88, 95],
})

print("Shape:", df.shape)
print(df.head())

analysis = bd.analyze_dataset(df)
print(analysis)

recommendations = bd.recommend_preprocessing(df)
print(recommendations)

profile = bd.profile_dataset(df)
print(profile)

charts = bd.create_visualizations(df)
print(charts)
```

Install pandas separately if you want to create DataFrames directly:

```bash
python -m pip install pandas
```

## Requirements

- Python 3.10 or newer
- pandas 2.0 or newer
- NumPy 1.24 or newer

Visualization features require the optional `matplotlib` and `seaborn` dependencies.

## Project Structure

```text
pybd/
├── pyproject.toml
├── README.md
├── src/
│   └── pybd_in/
└── tests/
```

## Testing

To run the test suite from the project directory:

```bash
python -m pip install -e ".[dev,visualization]"
python -m pytest -v
```

The project uses GitHub Actions to test supported Python versions across Linux, Windows, and macOS.

## Limitations

- Preprocessing recommendations are suggestions; review them before modifying your data.
- Results depend on the dataset and the functions implemented in the installed version.
- Visualization generation requires the optional visualization dependencies.
- This package does not replace domain knowledge or careful data validation.

## Contributing

Contributions, bug reports, and feature suggestions are welcome.

1. Fork the repository.
2. Create a branch for your changes.
3. Make your changes and add tests where appropriate.
4. Run the test suite.
5. Open a pull request.

## Bug Reports and Feature Requests

Report issues or request features through the GitHub issue tracker:

https://github.com/AnikDutta-AI/Pybd-Pypi-Pakage/issues

## Repository

https://github.com/AnikDutta-AI/Pybd-Pypi-Pakage

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

**Anik Dutta**

Email: anikdatta979@gmail.com
