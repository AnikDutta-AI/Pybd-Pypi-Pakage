# PYBD

**PYBD** is a Python package for dataset analysis, data quality assessment, and preprocessing recommendations.

## Features

- Load datasets into pandas DataFrames.
- Analyze dataset structure and quality.
- Identify missing values and potential data quality issues.
- Generate preprocessing recommendations.
- Create dataset visualizations.
- Generate dataset profile reports and export them as JSON.

## Installation

Install the package from PyPI once it has been published:

```bash
pip install pybd-in
```

To install visualization dependencies:

```bash
pip install "pybd-in[visualization]"
```

## Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/AnikDutta-AI/Pybd-Pypi-Pakage.git
cd Pybd-Pypi-Pakage
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install development dependencies

```bash
python -m pip install -e ".[dev,visualization]"
```

### 4. Run tests

```bash
python -m pytest -v
```

## Usage

```python
from pybd_in import (
    load_dataset,
    analyze_dataset,
    recommend_preprocessing,
    profile_dataset,
)

# Load a dataset
df = load_dataset("data.csv")

# Analyze the dataset
analysis = analyze_dataset(df)
print(analysis)

# Get preprocessing recommendations
recommendations = recommend_preprocessing(df)
print(recommendations)

# Generate a dataset profile
profile = profile_dataset("data.csv", generate_charts=False)
print(profile.render_report())

# Export the profile as JSON
print(profile.to_json())
```

For visualization support, install the optional visualization dependencies before generating charts.

## Requirements

- Python 3.10 or newer
- pandas
- NumPy

Visualization features additionally use Matplotlib and Seaborn.

## Project Structure

```text
src/
└── pybd_in/
    ├── __init__.py
    ├── dataset.py
    ├── analysis.py
    ├── preprocessing.py
    ├── visualization.py
    └── dataset_profiler.py
```

## Author

**Anik Dutta**  
Email: anikdatta@gmail.com

## License

Add your chosen open-source license here and include the corresponding `LICENSE` file in the repository.
