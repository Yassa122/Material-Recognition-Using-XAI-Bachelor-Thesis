import pandas as pd
import torch
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_chemberta")
model.eval()

# Load the predicted properties data
predicted_df = pd.read_csv("predicted_properties.csv")  # Ensure this file exists

# Ensure the SMILES column exists and is properly formatted
if "SMILES" not in predicted_df.columns:
    raise ValueError("The 'SMILES' column is missing in the input data.")

# Convert SMILES to a list of strings
smiles_data = predicted_df["SMILES"][
    :100
].tolist()  # Take the first 100 SMILES as a list


# Function for model predictions
def model_predict(smiles_list):
    # Ensure input is a list of strings
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]  # Convert a single string to a list

    # Tokenize input SMILES
    inputs = tokenizer(
        smiles_list, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Perform prediction
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.cpu().numpy()


# Initialize SHAP KernelExplainer
explainer = shap.KernelExplainer(
    model_predict, data=smiles_data[:10]
)  # Use only the first 10 for initialization

# Generate explanations for the first sample
smiles_sample = [smiles_data[0]]  # Take a single SMILES string for explanation
shap_values = explainer.shap_values(smiles_sample)

# Display SHAP values
print("SHAP Values for the sample:")
print(shap_values)

# Optional: Visualize the SHAP explanation
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], smiles_sample)
