import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rdkit import Chem
from rdkit.Chem import Draw

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_chemberta")
model.eval()

# Load the data with predictions
data = pd.read_csv("predicted_properties.csv")  # Ensure this path is correct


# Function to generate predictions
def predict_smiles(model, tokenizer, smiles_list):
    tokenized_inputs = tokenizer(
        smiles_list, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    inputs = {
        "input_ids": tokenized_inputs["input_ids"].to(model.device),
        "attention_mask": tokenized_inputs["attention_mask"].to(model.device),
    }
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.cpu().numpy()


# Function to compare original and counterfactual SMILES and generate an explanation
def explain_counterfactual(original_smiles, counterfactual_smiles, model, tokenizer):
    # Predict properties for original and counterfactual
    original_prediction = predict_smiles(model, tokenizer, [original_smiles])[0]
    counterfactual_prediction = predict_smiles(
        model, tokenizer, [counterfactual_smiles]
    )[0]

    # Parse molecules using RDKit
    original_mol = Chem.MolFromSmiles(original_smiles)
    counterfactual_mol = Chem.MolFromSmiles(counterfactual_smiles)

    # Visualize molecules (optional for Jupyter Notebooks or image output)
    if original_mol and counterfactual_mol:
        print("Original Molecule:")
        display(Draw.MolToImage(original_mol))
        print("Counterfactual Molecule:")
        display(Draw.MolToImage(counterfactual_mol))

    # Analyze structural changes
    differences = []
    for i, (orig_char, cf_char) in enumerate(
        zip(original_smiles, counterfactual_smiles)
    ):
        if orig_char != cf_char:
            differences.append((i, orig_char, cf_char))

    # Generate explanation
    explanation = f"Original SMILES: {original_smiles}\n"
    explanation += f"Original prediction: {original_prediction}\n"
    explanation += f"Counterfactual SMILES: {counterfactual_smiles}\n"
    explanation += f"Counterfactual prediction: {counterfactual_prediction}\n\n"
    explanation += "Differences observed:\n"

    for diff in differences:
        explanation += f" - Position {diff[0]}: '{diff[1]}' changed to '{diff[2]}'\n"

    explanation += "\nImpact on properties:\n"
    explanation += "The changes observed in the SMILES string could indicate:\n"
    explanation += " - Altered substructure affecting stability, reactivity, or binding affinity.\n"
    explanation += (
        " - Potential shifts in molecular polarity or functional group interactions.\n"
    )

    print(explanation)


# Select an example from your data and generate a counterfactual
original_smiles = data["SMILES"].iloc[
    0
]  # Take the first SMILES from your data for analysis
counterfactual_smiles = "68C1C2CCC(C2)C1CN(CCO)C(=O)c1ccc(Cl)cc1"  # Replace this with a generated or selected counterfactual

# Run the counterfactual explanation
explain_counterfactual(original_smiles, counterfactual_smiles, model, tokenizer)
