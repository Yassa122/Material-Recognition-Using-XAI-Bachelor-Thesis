import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load the CSV files
math_based_file = (
    "math_based_properties.csv"  # Replace with your math-based CSV filename
)
predicted_file = (
    "predicted_properties.csv"  # Replace with your transformer predictions CSV filename
)

# Load the dataframes
math_based_df = pd.read_csv(math_based_file)
predicted_df = pd.read_csv(predicted_file)

# Print column names to verify
print("Math-based DataFrame columns:", math_based_df.columns)
print("Predicted DataFrame columns:", predicted_df.columns)

# Rename columns to align with property comparison (adjust these as needed)
property_mapping = {
    "Property_1": "MolWt",  # Update with the appropriate column from math-based DataFrame
    "Property_2": "NumAtoms",  # Update as needed
    "Property_3": "LogP",  # Update as needed
}

# Iterate through properties and calculate metrics
for pred_col, math_col in property_mapping.items():
    # Check if the columns exist in the DataFrames
    if math_col not in math_based_df.columns or pred_col not in predicted_df.columns:
        print(f"Column {math_col} or {pred_col} not found in DataFrames.")
        continue

    # Calculate metrics
    mse = mean_squared_error(math_based_df[math_col], predicted_df[pred_col])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(math_based_df[math_col], predicted_df[pred_col])

    # Print metrics for each property
    print(f"Metrics for {math_col} (compared with {pred_col}):")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print()

# Overall metrics across all properties if needed
overall_mse = mean_squared_error(
    math_based_df[list(property_mapping.values())],
    predicted_df[list(property_mapping.keys())],
    multioutput="uniform_average",
)
overall_rmse = np.sqrt(overall_mse)
overall_mae = mean_absolute_error(
    math_based_df[list(property_mapping.values())],
    predicted_df[list(property_mapping.keys())],
    multioutput="uniform_average",
)

print("Overall Metrics:")
print(f"  Overall MSE: {overall_mse:.4f}")
print(f"  Overall RMSE: {overall_rmse:.4f}")
print(f"  Overall MAE: {overall_mae:.4f}")
