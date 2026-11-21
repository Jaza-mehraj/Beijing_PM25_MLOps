
from pathlib import Path
import pandas as pd

# Define the project root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the CSV file path
CSV_PATH = PROJECT_ROOT / "data" / "raw" / (
    "PRSA_Data_Aotizhongxin_20130301-20170228.csv"
)

# Check whether the dataset exists
if not CSV_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

# Load the dataset
df = pd.read_csv(CSV_PATH)

# Display dataset information
print("DATASET LOADED")
print("Rows and columns:", df.shape)
print("Column names:", df.columns.tolist())

# Display first five rows
print("First five rows:")
print(df.head())

# Display data types
print("\nDATA TYPES")
print(df.dtypes)

# Display missing values per column
print("\nMISSING VALUES PER COLUMN")
missing_counts = df.isna().sum()
print(missing_counts)

# Display total missing cells
print("\nTOTAL MISSING CELLS")
print(missing_counts.sum())

# Display PM2.5 summary statistics
print("\nPM2.5 SUMMARY")
print(df["PM2.5"].describe())

# Count duplicate rows
print("\nNUMBER OF DUPLICATE ROWS")
print(df.duplicated().sum())

# Create timestamp column
df["timestamp"] = pd.to_datetime(
    df[["year", "month", "day", "hour"]]
)

# Display time range information
print("\nTIME RANGE")
print("First timestamp:", df["timestamp"].min())
print("Last timestamp:", df["timestamp"].max())
print(
    "Timestamps sorted:",
    df["timestamp"].is_monotonic_increasing
)

# Define the inspection report file path
report_path = PROJECT_ROOT / "reports" / "inspection_report.txt"

# Write dataset inspection results to a text file
with report_path.open("w", encoding="utf-8") as report:
    report.write(f"Shape: {df.shape}\n")

    report.write("Missing values by column:\n")
    report.write(missing_counts.to_string())

    report.write("\n\nPM2.5 summary:\n")
    report.write(df["PM2.5"].describe().to_string())

    report.write("\n")

# Confirm report creation
print("\nINSPECTION REPORT SAVED")
print(report_path)
