import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load math-based (ground truth) and transformer-predicted data
math_based_file = "math_based_properties.csv"  # CSV file with ground truth values
predicted_file = "predicted_properties.csv"  # CSV file with transformer predictions

# Load the data
math_based_df = pd.read_csv(math_based_file)
predicted_df = pd.read_csv(predicted_file)

# Check if required columns are present in the DataFrames
if (
    "NumAtoms" not in math_based_df.columns
    or "Predicted_logP" not in predicted_df.columns
):
    print("NumAtoms or Property_2 column is missing.")
else:
    # Metrics for Property_2 (NumAtoms)
    mae_property_2 = mean_absolute_error(
        math_based_df["NumAtoms"], predicted_df["Predicted_logP"]
    )
    mse_property_2 = mean_squared_error(
        math_based_df["NumAtoms"], predicted_df["Predicted_logP"]
    )
    rmse_property_2 = np.sqrt(mse_property_2)

    print("Metrics for Property_2 (NumAtoms):")
    print(f"  MAE: {mae_property_2:.4f}")
    print(f"  MSE: {mse_property_2:.4f}")
    print(f"  RMSE: {rmse_property_2:.4f}\n")

if (
    "LogP" not in math_based_df.columns
    or "Predicted_num_atoms" not in predicted_df.columns
):
    print("LogP or Property_3 column is missing.")
else:
    # Metrics for Property_3 (LogP)
    mae_property_3 = mean_absolute_error(
        math_based_df["LogP"], predicted_df["Predicted_num_atoms"]
    )
    mse_property_3 = mean_squared_error(
        math_based_df["LogP"], predicted_df["Predicted_num_atoms"]
    )
    rmse_property_3 = np.sqrt(mse_property_3)

    print("Metrics for Property_3 (LogP):")
    print(f"  MAE: {mae_property_3:.4f}")
    print(f"  MSE: {mse_property_3:.4f}")
    print(f"  RMSE: {rmse_property_3:.4f}\n")

# Overall metrics if you want an average across multiple properties
overall_mae = mean_absolute_error(
    math_based_df[["NumAtoms", "LogP"]],
    predicted_df[["Predicted_logP", "Predicted_num_atoms"]],
)
overall_mse = mean_squared_error(
    math_based_df[["NumAtoms", "LogP"]],
    predicted_df[["Predicted_logP", "Predicted_num_atoms"]],
)
overall_rmse = np.sqrt(overall_mse)

print("Overall Metrics for Property_2 and Property_3:")
print(f"  Overall MAE: {overall_mae:.4f}")
print(f"  Overall MSE: {overall_mse:.4f}")
print(f"  Overall RMSE: {overall_rmse:.4f}")


# Calculate R-squared for each property
r2_property_2 = r2_score(math_based_df["NumAtoms"], predicted_df["Predicted_logP"])
r2_property_3 = r2_score(math_based_df["LogP"], predicted_df["Predicted_num_atoms"])

# Calculate an average R-squared score as an overall accuracy metric
overall_r2 = (r2_property_2 + r2_property_3) / 2

print(f"R-squared for Property_2 (NumAtoms): {r2_property_2:.4f}")
print(f"R-squared for Property_3 (LogP): {r2_property_3:.4f}")
print(f"Overall R-squared (Accuracy from 0 to 1): {overall_r2:.4f}")

# Scatter plot for Property_2 (NumAtoms)
plt.figure(figsize=(8, 6))
plt.scatter(
    math_based_df["NumAtoms"],
    predicted_df["Property_2"],
    alpha=0.5,
    label="Predicted vs Math-Based",
)
plt.plot(
    [math_based_df["NumAtoms"].min(), math_based_df["NumAtoms"].max()],
    [math_based_df["NumAtoms"].min(), math_based_df["NumAtoms"].max()],
    "r--",
    label="Perfect Prediction Line",
)
plt.xlabel("Math-Based NumAtoms")
plt.ylabel("Transformer-Predicted Property_2")
plt.title("Comparison of Math-Based NumAtoms and Transformer-Predicted Property_2")
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot for Property_3 (LogP)
plt.figure(figsize=(8, 6))
plt.scatter(
    math_based_df["LogP"],
    predicted_df["Property_3"],
    alpha=0.5,
    label="Predicted vs Math-Based",
)
plt.plot(
    [math_based_df["LogP"].min(), math_based_df["LogP"].max()],
    [math_based_df["LogP"].min(), math_based_df["LogP"].max()],
    "r--",
    label="Perfect Prediction Line",
)
plt.xlabel("Math-Based LogP")
plt.ylabel("Transformer-Predicted Property_3")
plt.title("Comparison of Math-Based LogP and Transformer-Predicted Property_3")
plt.legend()
plt.grid(True)
plt.show()
