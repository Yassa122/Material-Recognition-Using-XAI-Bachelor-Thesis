# preprocess_and_train.py

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


def calculate_descriptors(smiles):
    """
    Calculate molecular descriptors using RDKit.

    Parameters:
        smiles (str): SMILES string of the compound.

    Returns:
        dict: Dictionary containing MolWt, NumHDonors, NumHAcceptors, logP.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            "MolWt": Descriptors.MolWt(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "logP": Crippen.MolLogP(mol),
        }
    else:
        return {
            "MolWt": np.nan,
            "NumHDonors": np.nan,
            "NumHAcceptors": np.nan,
            "logP": np.nan,
        }


def preprocess_data(csv_path):
    """
    Load and preprocess binding affinity data.

    Parameters:
        csv_path (str): Path to the binding affinity CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with descriptors and pIC50.
    """
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {len(df)} records from {csv_path}.")

    # Calculate descriptors
    logging.info("Calculating molecular descriptors...")
    descriptors = df["SMILES"].apply(calculate_descriptors).apply(pd.Series)
    df = pd.concat([df, descriptors], axis=1)

    # Drop rows with missing descriptors or pIC50
    initial_len = len(df)
    df.dropna(
        subset=["MolWt", "NumHDonors", "NumHAcceptors", "logP", "pIC50"], inplace=True
    )
    final_len = len(df)
    logging.info(f"Dropped {initial_len - final_len} records due to missing values.")

    return df


def train_model(df):
    """
    Train a Random Forest regression model to predict pIC50.

    Parameters:
        df (pd.DataFrame): Preprocessed DataFrame.

    Returns:
        RandomForestRegressor: Trained regression model.
        dict: Model evaluation metrics.
    """
    # Features and target
    X = df[["MolWt", "NumHDonors", "NumHAcceptors", "logP"]]
    y = df["pIC50"]

    # Split the data
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize the model
    logging.info("Initializing the Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    logging.info("Training the model...")
    rf_model.fit(X_train, y_train)

    # Evaluate the model
    logging.info("Evaluating the model...")
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Model Evaluation - MSE: {mse:.4f}, R²: {r2:.4f}")

    metrics = {"MSE": mse, "R2": r2}

    return rf_model, metrics


def save_model(model, model_path):
    """
    Save the trained model using joblib.

    Parameters:
        model (RandomForestRegressor): Trained regression model.
        model_path (str): Path to save the model.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}.")


if __name__ == "__main__":
    # Paths
    data_csv = "binding_affinity_data.csv"  # Ensure this file exists
    model_save_path = "./models/binding_affinity_rf_model.joblib"

    # Preprocess data
    df_processed = preprocess_data(data_csv)

    # Train model
    model, evaluation_metrics = train_model(df_processed)

    # Save model
    save_model(model, model_save_path)

    # Optionally, save the evaluation metrics
    metrics_df = pd.DataFrame([evaluation_metrics])
    metrics_df.to_csv("./models/model_evaluation_metrics.csv", index=False)
    logging.info("Training and saving completed successfully.")
