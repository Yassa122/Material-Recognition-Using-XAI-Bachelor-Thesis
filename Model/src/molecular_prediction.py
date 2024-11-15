import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import shap

# Load dataset
data = pd.read_csv("test.csv")

# Drop rows where pIC50, logP, or num_atoms are NaN
data = data.dropna(subset=["pIC50", "logP", "num_atoms"])

# Keep a copy of the original data to merge later
original_data = data.copy()

# Drop non-numeric columns like 'SMILES' and 'mol'
# Features are everything except 'SMILES', 'mol', and target columns ('pIC50', 'logP', 'num_atoms')
X = data.drop(columns=["pIC50", "logP", "num_atoms", "SMILES", "mol"])

# Target columns (you can add more chemical descriptors if necessary)
y = data[["pIC50", "logP", "num_atoms"]]  # Predicting pIC50, logP, and num_atoms

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the RandomForest model with multi-output capability
model = MultiOutputRegressor(RandomForestRegressor())
model.fit(X_train, y_train)

# Predict the chemical properties (pIC50, logP, num_atoms) on the test set
y_pred = model.predict(X_test)

# Create a DataFrame with predictions
predictions_df = X_test.copy()
predictions_df[["predicted_pIC50", "predicted_logP", "predicted_num_atoms"]] = (
    y_pred  # Add the predicted columns
)

# Combine the original data with the predictions
combined_df = pd.concat(
    [
        original_data.loc[predictions_df.index],
        predictions_df[["predicted_pIC50", "predicted_logP", "predicted_num_atoms"]],
    ],
    axis=1,
)

# Save the combined dataset (original data + predictions) to CSV
combined_df.to_csv("tested_combined_molecular_data_with_predictions.csv", index=False)

# Explain predictions with SHAP for pIC50, logP, and num_atoms
explainer_pIC50 = shap.TreeExplainer(
    model.estimators_[0]
)  # For the first target (pIC50)
shap_values_pIC50 = explainer_pIC50.shap_values(X_test)

explainer_logP = shap.TreeExplainer(
    model.estimators_[1]
)  # For the second target (logP)
shap_values_logP = explainer_logP.shap_values(X_test)

explainer_num_atoms = shap.TreeExplainer(
    model.estimators_[2]
)  # For the third target (num_atoms)
shap_values_num_atoms = explainer_num_atoms.shap_values(X_test)

# Plot SHAP summary for pIC50
shap.summary_plot(shap_values_pIC50, X_test, feature_names=X.columns)

# Plot SHAP summary for logP
shap.summary_plot(shap_values_logP, X_test, feature_names=X.columns)

# Plot SHAP summary for num_atoms
shap.summary_plot(shap_values_num_atoms, X_test, feature_names=X.columns)
