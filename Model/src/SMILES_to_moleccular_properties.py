from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

# Example dataset with SMILES strings
data = pd.read_csv("test.csv")  # Replace "test.csv" with your dataset file

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
        # You can add more descriptors as needed
    }

# Apply the function to generate descriptors for each SMILES string
descriptors_list = data["SMILES"].apply(calculate_descriptors)

# Convert the descriptors into a DataFrame
descriptors_df = pd.DataFrame(descriptors_list.tolist())

# Combine the original data with the descriptors
combined_data = pd.concat([data, descriptors_df], axis=1)

# Save the combined dataset to CSV
combined_data.to_csv("tested_combined_molecular_data.csv", index=False)
