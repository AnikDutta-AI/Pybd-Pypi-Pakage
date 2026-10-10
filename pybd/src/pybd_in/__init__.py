
"""
pybd - Dataset analysis, data quality assessment,
and preprocessing recommendations.
"""

__version__ = "0.0.1"
__author__ = "Anik Dutta"
__email__ = "anikdatta@gmail.com"

from pybd_in.analysis import analyze_dataset
from pybd_in.dataset import load_dataset
from pybd_in.preprocessing import recommend_preprocessing
from pybd_in.dataset_profiler import DatasetProfile, profile_dataset

__all__ = [
    "load_dataset",
    "analyze_dataset",
    "recommend_preprocessing",
    "DatasetProfile",
    "profile_dataset",
    "create_visualizations",
]


def create_visualizations(*args, **kwargs):
    """Generate dataset visualizations using optional dependencies."""
    from pybd_in.visualization import (
        create_visualizations as _create_visualizations,
    )

    return _create_visualizations(*args, **kwargs)
