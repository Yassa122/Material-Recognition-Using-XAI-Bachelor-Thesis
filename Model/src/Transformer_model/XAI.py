import pandas as pd
import torch
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_chemberta")
model.eval()

# Load the predicted properties data
predicted_df = pd.read_csv("predicted_properties.csv")

# Ensure the SMILES column exists and is properly formatted
if "SMILES" not in predicted_df.columns:
    raise ValueError("The 'SMILES' column is missing in the input data.")

# Convert SMILES to a list of strings
smiles_data = predicted_df["SMILES"].tolist()

# Define a fixed max length for tokenization
max_length = 128

# Tokenize the first 10 SMILES for SHAP initialization
tokenized_data = tokenizer(
    smiles_data[:10],
    padding="max_length",
    truncation=True,
    max_length=max_length,
    return_tensors="pt",
)
tokenized_input_ids = tokenized_data["input_ids"].numpy()  # Convert to NumPy array


# Prediction function for SHAP
def model_predict(tokenized_inputs):
    inputs = {
        "input_ids": torch.tensor(tokenized_inputs).to(model.device),
        "attention_mask": torch.ones_like(torch.tensor(tokenized_inputs)).to(
            model.device
        ),
    }
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.cpu().numpy()


# Initialize SHAP KernelExplainer
explainer = shap.KernelExplainer(
    model_predict, data=tokenized_input_ids[:1]
)  # Use a single example for consistent input shape

# Tokenize a single sample SMILES string with fixed max length for prediction
sample_smiles = smiles_data[0]
sample_tokenized = tokenizer(
    [sample_smiles],
    padding="max_length",
    truncation=True,
    max_length=max_length,
    return_tensors="pt",
)["input_ids"].numpy()

# Generate SHAP values
shap_values = explainer.shap_values(sample_tokenized)

# Display SHAP values for Property_1, Property_2, and Property_3
print("SHAP Values for Property_1, Property_2, and Property_3 for the sample:")
print(shap_values)

# Optional: Visualize the SHAP explanation
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0][0], sample_smiles)
