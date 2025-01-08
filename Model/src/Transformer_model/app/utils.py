# app/utils.py

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def compute_additional_descriptors(mol, rdkit_descriptors):
    vals = []
    for desc_name in rdkit_descriptors:
        func = getattr(Descriptors, desc_name)
        try:
            val = func(mol)
        except Exception:
            val = 0  # Handle descriptors that might fail
        vals.append(val)
    return vals


def extract_knowledge_features(smiles, rdkit_descriptors):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        molecular_weight = Descriptors.MolWt(mol)
        num_h_donors = Descriptors.NumHDonors(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        additional_vals = compute_additional_descriptors(mol, rdkit_descriptors)
        return [molecular_weight, num_h_donors, num_h_acceptors] + additional_vals
    else:
        return [0, 0, 0] + [0] * len(rdkit_descriptors)


def load_smiles_data(
    csv_file, rdkit_descriptors, properties_present=True, fit_scaler=False, scalers=None
):
    data = pd.read_csv(csv_file)
    data = data.dropna(subset=["SMILES"])
    smiles = data["SMILES"].tolist()
    knowledge_features = [
        extract_knowledge_features(sm, rdkit_descriptors) for sm in smiles
    ]

    if properties_present:
        properties = ["pIC50", "logP", "num_atoms"]
        targets = data[properties].apply(pd.to_numeric, errors="coerce")
        valid_indices = targets.dropna().index
        smiles = [smiles[i] for i in valid_indices]
        knowledge_features = [knowledge_features[i] for i in valid_indices]
        targets = targets.loc[valid_indices]

        if not np.isfinite(targets).all().all():
            logging.error("Infinite values detected in targets. Please clean the data.")
            raise ValueError("Infinite values detected in targets.")

        if fit_scaler and scalers:
            scalers["scaler_pIC50"].fit(targets["pIC50"].values.reshape(-1, 1))
            scalers["scaler_logP"].fit(targets["logP"].values.reshape(-1, 1))
            scalers["scaler_num_atoms"].fit(targets["num_atoms"].values.reshape(-1, 1))
            logging.info("Individual scalers fitted on training data.")

        pIC50_scaled = scalers["scaler_pIC50"].transform(
            targets["pIC50"].values.reshape(-1, 1)
        )
        logP_scaled = scalers["scaler_logP"].transform(
            targets["logP"].values.reshape(-1, 1)
        )
        num_atoms_scaled = scalers["scaler_num_atoms"].transform(
            targets["num_atoms"].values.reshape(-1, 1)
        )

        targets_scaled = np.hstack([pIC50_scaled, logP_scaled, num_atoms_scaled])
        return smiles, targets_scaled, knowledge_features
    else:
        return smiles, None, knowledge_features
