import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import os


# Function to calculate descriptors
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return [
            Descriptors.MolWt(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Crippen.MolLogP(mol),
        ]
    else:
        return [np.nan, np.nan, np.nan, np.nan]


# Load your binding affinity dataset
df_binding = pd.read_csv("binding_affinity_data.csv")  # Replace with your dataset path

# Calculate descriptors
descriptor_cols = ["MolWt", "NumHDonors", "NumHAcceptors", "logP"]
df_binding[descriptor_cols] = df_binding["SMILES"].apply(
    lambda x: pd.Series(calculate_descriptors(x))
)

# Drop rows with invalid SMILES or missing descriptors
df_binding.dropna(subset=descriptor_cols + ["pIC50"], inplace=True)

# Features and target
X = df_binding[descriptor_cols]
y = df_binding["pIC50"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Random Forest Regressor - MSE: {mse}, R²: {r2}")

# Save the trained model
model_dir = "./models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "binding_affinity_rf_model.joblib")
joblib.dump(rf_model, model_path)
print(f"Model saved to {model_path}")
