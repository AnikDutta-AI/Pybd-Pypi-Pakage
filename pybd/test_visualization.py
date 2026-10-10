# import pandas as pd
#
# from pybd_in.visualization import (
#     VisualizationConfig,
#     create_visualizations,
# )
#
# df = pd.DataFrame({
#     "age": [22, 25, 30, 35, 40],
#     "salary": [25000, 35000, 42000, 50000, 65000],
#     "department": ["IT", "HR", "IT", "Sales", "HR"],
#     "rating": [4.0, 3.5, 4.5, 4.0, 5.0],
# })
#
# config = VisualizationConfig(
#     max_numeric_columns=10,
#     max_categorical_columns=10,
#     max_categories=15,
#     histogram_bins="auto",
#     dpi=150,
#     image_format="png",
# )
#
# charts = create_visualizations(
#     df,
#     output_dir="pybd_report/charts",
#     config=config,
# )
#
# print(f"Generated {len(charts)} charts.")
#
# for chart in charts:
#     print(chart)
#
import sys
import pandas as pd
from pathlib import Path
from IPython.display import display, Image

# Locate your package source in Colab
sys.path.insert(0, "/content/pybd_upload/pybd/src")

from pybd_in.visualization import (
    VisualizationConfig,
    create_visualizations,
)

# Sample dataset
df = pd.DataFrame({
    "age": [22, 25, 30, 35, 40],
    "salary": [25000, 35000, 42000, 50000, 65000],
    "department": ["IT", "HR", "IT", "Sales", "HR"],
    "rating": [4.0, 3.5, 4.5, 4.0, 5.0],
})

print("Dataset preview:")
display(df)

# Visualization configuration
config = VisualizationConfig(
    max_numeric_columns=10,
    max_categorical_columns=10,
    max_categories=15,
    histogram_bins="auto",
    dpi=150,
    image_format="png",
)

# Generate charts
output_dir = "/content/pybd_report/charts"

charts = create_visualizations(
    df,
    output_dir=output_dir,
    config=config,
)

print(f"\nGenerated {len(charts)} charts.")

# Display each chart
for chart in charts:
    print(f"\nChart: {Path(chart).name}")
    display(Image(filename=str(chart)))