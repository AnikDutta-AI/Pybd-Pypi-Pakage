
from pathlib import Path
import json
import pandas as pd

from pybd_in.dataset_profiler import profile_dataset

# 1. Create sample data directly in Python
df = pd.DataFrame({
    "age": [22, 25, 30, 35, 40, 30],
    "salary": [25000, 35000, 42000, 50000, 65000, None],
    "department": ["IT", "HR", "IT", "Sales", "HR", "IT"],
    "rating": [4.0, 3.5, 4.5, 4.0, 5.0, 3.0],
})

# 2. Save the DataFrame as a CSV file automatically
test_dir = Path("test_output")
test_dir.mkdir(parents=True, exist_ok=True)

csv_path = test_dir / "sample.csv"
df.to_csv(csv_path, index=False)

print("Sample dataset created:", csv_path.resolve())

# 3. Test profiling WITH charts
profile = profile_dataset(
    file_path=csv_path,
    output_dir=test_dir / "charts_enabled",
    generate_charts=True,
)

print("\n--- PROFILE WITH CHARTS ---")
print("Profile:", profile)
print("Dataset shape:", profile.shape)
print("Critical issues:", profile.has_critical_issues)

profile.show_report()

print("\nGenerated chart files:")
for chart in profile.charts:
    print(chart, "Exists:", chart.exists())

# 4. Test profiling WITHOUT charts
profile_no_charts = profile_dataset(
    file_path=csv_path,
    output_dir=test_dir / "charts_disabled",
    generate_charts=False,
)

print("\n--- PROFILE WITHOUT CHARTS ---")
print("Dataset shape:", profile_no_charts.shape)
print("Chart count:", len(profile_no_charts.charts))

assert profile_no_charts.shape == (6, 4)
assert len(profile_no_charts.charts) == 0
assert profile_no_charts.charts_skipped_reason is None

print("PASS: Profiling works without charts.")

# 5. Test JSON serialization
json_text = profile_no_charts.to_json()
parsed_json = json.loads(json_text)

json_path = test_dir / "profile_report.json"
json_path.write_text(json_text, encoding="utf-8")

assert isinstance(parsed_json, dict)
assert json_path.exists()

print("\nJSON valid:", isinstance(parsed_json, dict))
print("JSON report saved:", json_path.resolve())
print("PASS: JSON serialization works.")
