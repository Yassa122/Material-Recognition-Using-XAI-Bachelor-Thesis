from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import shap


# Function to calculate molecular descriptors from SMILES
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAtoms": mol.GetNumAtoms(),
    }


# Load the combined dataset
combined_data = pd.read_csv("tested_combined_molecular_data.csv")

# Drop rows with missing values
combined_data = combined_data.dropna()

# Separate features (descriptors) and target (chemical properties)
X = combined_data[["MolWt", "LogP", "NumRotatableBonds", "NumAtoms"]]
y = combined_data[
    ["MolWt", "LogP", "NumRotatableBonds", "NumAtoms"]
]  # Target properties

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the RandomForest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the performance (e.g., with Mean Squared Error or R^2 Score)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Load new SMILES data
new_smiles_data = pd.read_csv(
    "tested_combined_molecular_data.csv"
)  # Replace with your new SMILES dataset

# Generate descriptors for new SMILES strings
new_descriptors_list = new_smiles_data["SMILES"].apply(calculate_descriptors)
new_descriptors_df = pd.DataFrame(new_descriptors_list.tolist())

# Predict the chemical properties using the trained model
new_predictions = model.predict(new_descriptors_df)

# Add the predicted properties to the original dataset
new_smiles_data["Predicted_MolWt"] = new_predictions[:, 0]  # Predicted molecular weight
new_smiles_data["Predicted_LogP"] = new_predictions[:, 1]  # Predicted LogP
new_smiles_data["Predicted_NumRotatableBonds"] = new_predictions[
    :, 2
]  # Predicted number of rotatable bonds
new_smiles_data["Predicted_NumAtoms"] = new_predictions[
    :, 3
]  # Predicted number of atoms

# Save the predictions to a CSV file
new_smiles_data.to_csv("predicted_molecular_properties.csv", index=False)

# Residuals for Molecular Weight (MolWt)
residuals_MolWt = y_test["MolWt"] - y_pred[:, 0]
plt.hist(residuals_MolWt, bins=30, color="blue")
plt.xlabel("Residuals (MolWt)")
plt.ylabel("Frequency")
plt.title("Residuals Distribution for Molecular Weight")
plt.show()

# Residuals for LogP
residuals_LogP = y_test["LogP"] - y_pred[:, 1]
plt.hist(residuals_LogP, bins=30, color="green")
plt.xlabel("Residuals (LogP)")
plt.ylabel("Frequency")
plt.title("Residuals Distribution for LogP")
plt.show()

# 5-fold cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
print("Cross-Validation R² Scores:", cv_scores)
print("Average Cross-Validation R²:", cv_scores.mean())

# SHAP Explainer and Visualization
# Initialize SHAP explainer for the trained RandomForest model
explainer = shap.Explainer(model, X_train)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# Summary plot for feature importance (make it like your screenshot)
shap.summary_plot(shap_values, X_test, plot_type="dot")

# Dependence plot for individual features (e.g., LogP)
shap.dependence_plot("LogP", shap_values.values, X_test)

# Force plot for a specific prediction (e.g., the first instance in the test set)
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0, :])
