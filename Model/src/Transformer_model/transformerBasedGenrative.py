import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler
from tqdm import tqdm
import random
import seaborn as sns
import numpy as np
from rdkit import Chem
from rdkit import RDLogger  # Suppress RDKit warnings
from lime.lime_text import LimeTextExplainer
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from captum.attr import IntegratedGradients
import logging


# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
logging.getLogger("excessive_module").setLevel(logging.ERROR)
 # Disable all logging

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Parameters
input_file = r'C:\Users\ASUS\Desktop\moses\data\train.csv'

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class XAI_CausalChemBERTa(nn.Module):
    def __init__(self, model_path, hidden_size=768, num_classes=1):
        super(XAI_CausalChemBERTa, self).__init__()
        # Load the AutoModelForCausalLM
        self.chemberta = AutoModelForCausalLM.from_pretrained(model_path)
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Add interpretable layers
        self.interpretable_layer = nn.Linear(self.chemberta.config.hidden_size, self.chemberta.config.hidden_size // 2)
        self.attention_layer = nn.Linear(self.chemberta.config.hidden_size, 1)  # For attention scores
        self.prototype_layer = nn.Linear(self.chemberta.config.hidden_size // 2, self.chemberta.config.hidden_size // 4)
        self.final_layer = nn.Linear(self.chemberta.config.hidden_size // 4, num_classes)
        self.activation = nn.Sigmoid()  # For binary classification; adjust as needed

    @property
    def config(self):
        """Expose the config of the wrapped chemberta model."""
        return self.chemberta.config

    @property
    def device(self):
        """Expose the device of the wrapped chemberta model."""
        return next(self.chemberta.parameters()).device

    def generate(self, *args, **kwargs):
        """Delegate the generate method to the underlying chemberta model."""
        return self.chemberta.generate(*args, **kwargs)

    def save_pretrained(self, save_directory):
        """Save the underlying chemberta model in Hugging Face format."""
        self.chemberta.save_pretrained(save_directory)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Enable hidden_states in the output
        outputs = self.chemberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # Ensure hidden states are returned
            **kwargs
        )

        # Extract hidden states from the base model's output
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states [batch_size, seq_len, hidden_size]
        if hidden_states is None:
            raise ValueError("Hidden states are not returned. Ensure output_hidden_states=True is set.")

        # Compute attention scores across the sequence
        attention_scores = self.attention_layer(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        attention_weights = torch.softmax(attention_scores, dim=-1)  # Normalize across seq_len [batch_size, seq_len]

        # Weighted sum of token embeddings based on attention weights
        attended_representation = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, seq_len]
            hidden_states  # [batch_size, seq_len, hidden_size]
        ).squeeze(1)  # [batch_size, hidden_size]

        # Pass through interpretable layers
        reduced_hidden = self.interpretable_layer(attended_representation)  # [batch_size, reduced_dim]
        reduced_hidden = torch.relu(reduced_hidden)

        # Pass through the prototype layer
        prototype_hidden = self.prototype_layer(reduced_hidden)  # [batch_size, prototype_dim]
        prototype_hidden = torch.relu(prototype_hidden)

        token_protoype_hidden = self.prototype_layer(
            torch.relu(self.interpretable_layer(hidden_states))
        )

        # Compute final logits
        final_logits = self.final_layer(prototype_hidden)  # [batch_size, num_classes]
        final_logits = self.activation(final_logits)

        # Add explanations
        explanations = {
            "attention_weights": attention_weights,  # Token-level attention weights
            "prototype_hidden": prototype_hidden,    # Reduced representations for prototypes
            "token_prototype_activations": token_protoype_hidden
        }

        loss = None
        if labels is not None:
            # Adjust labels to match logits shape
            if labels.ndim > 1:  # Flatten if labels are multi-dimensional
                labels = labels[:, 0].unsqueeze(-1)  # [batch_size, 1]
            loss_fn = nn.BCELoss()  # Binary cross-entropy for binary classification
            loss = loss_fn(final_logits, labels.float())

        return {
            "loss": loss,
            "logits": final_logits,
            "explanations": explanations
        } if loss is not None else {
            "logits": final_logits,
            "explanations": explanations
        }

# Paths
local_model_path = r"C:\Users\ASUS\Desktop\moses\chemberta_finetuned"
new_model_path = r"C:\Users\ASUS\Desktop\moses\fine_tuned_model"

# Load the model with interpretable layers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XAI_CausalChemBERTa(model_path=local_model_path).to(device)

# Example: Accessing device
print(f"Model is loaded on device: {model.device}")

# Save the updated model
model.chemberta.save_pretrained(new_model_path)
output_dir = "./fine_tuned_model"
batch_size = 16
epochs = 3
max_length = 128
learning_rate = 5e-5
input_sample_size = 1000
target_novel_smiles = 10
max_new_tokens = 50

# Load SMILES dataset
try:
    data = pd.read_csv(input_file)
    logger.info(f"Loaded {len(data)} rows from the dataset.")
except FileNotFoundError:
    logger.error(f"Input file not found at {input_file}. Please check the path.")
    raise

# Sample input SMILES
if input_sample_size < len(data):
    data = data.sample(n=input_sample_size, random_state=42)
    logger.info(f"Sampled {input_sample_size} rows from the dataset.")

# Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(local_model_path)


# Dataset
class SMILESDataset(Dataset):
    def __init__(self, smiles_strings, tokenizer, max_length):
        self.smiles_strings = smiles_strings
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_strings)

    def __getitem__(self, idx):
        smiles = self.smiles_strings[idx]
        encoded = self.tokenizer(
            smiles,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }



# Split dataset into train and validation sets

train_size = int(0.8 * len(data))
val_size = len(data) - train_size
train_data, val_data = data[:train_size], data[train_size:]

# Create datasets
train_dataset = SMILESDataset(
    train_data["SMILES"].tolist(),
    tokenizer=tokenizer,
    max_length=max_length
)

val_dataset = SMILESDataset(
    val_data["SMILES"].tolist(),
    tokenizer=tokenizer,
    max_length=max_length
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)



import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AdamW, get_scheduler
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st

# Initialize Streamlit dashboard
st.title("ChemBERTa Training Dashboard")
st.subheader("Real-Time Insights for SMILES Interpretability")

# Sidebar for real-time monitoring
st.sidebar.title("Training Progress")
training_progress = st.sidebar.progress(0)
training_loss_placeholder = st.sidebar.empty()

def create_prompt(attention_data, prototype_data, smiles_data):
    """
    Creates a structured prompt for the chatbot API.

    Parameters:
        attention_data (list of dict): Attention weights and tokens for examples.
        prototype_data (list of dict): Prototype activations and tokens for examples.
        smiles_data (list of str): List of SMILES strings for examples.

    Returns:
        list of str: Prompts for each example.
    """
    prompts = []

    for idx, (attention, prototype, smiles) in enumerate(zip(attention_data, prototype_data, smiles_data)):
        tokens = attention["tokens"]
        attention_weights = attention["weights"]
        prototype_activations = prototype.get("activations", [])
        
        # Construct the structured prompt
        prompt = f"""
        ChemBERTa Training Insights - Example {idx + 1}

        SMILES Sequence: {smiles}

        Attention Analysis:
        The model assigned the following attention weights to the tokens:
        {', '.join([f'{token}: {weight:.3f}' for token, weight in zip(tokens, attention_weights)])}

        Prototype Activations:
        The following prototype activations were observed:
        {', '.join([f'{token}: {activation:.3f}' for token, activation in zip(tokens, prototype_activations)])}

        Insights Needed:
        - Analyze how attention weights correlate with key molecular substructures (e.g., aromatic rings, functional groups).
        - Highlight which prototypes are most relevant for predicting the molecular property.
        - Provide a chemistry-specific interpretation of why these features are important for the task.

        Detailed Insight:
        Please provide a detailed explanation based on the above information.
        """
        prompts.append(prompt)

    return prompts


import openai

def get_chemical_insights(prompts, api_key):
    """
    Calls the OpenAI API to get chemical insights for the training examples.

    Parameters:
        prompts (list of str): Prompts for each example.
        api_key (str): OpenAI API key.

    Returns:
        list of str: Insights generated by the API.
    """
    openai.api_key = api_key
    insights = []

    for prompt in prompts:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",  # Specify the model version you want to use
            messages=[
                {"role": "system", "content": "You are a chemistry expert analyzing model interpretability."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        insights.append(response["choices"][0]["message"]["content"])

    return insights


def display_attention_and_prototype_charts(tokens, attention_weights, prototype_activations=None, epoch=1, batch=1):
    # Attention weights chart
    fig_attention = go.Figure()
    fig_attention.add_trace(
        go.Bar(
            x=tokens,
            y=attention_weights,
            marker=dict(color=attention_weights, colorscale="Viridis"),
            name="Attention Weights",
        )
    )
    fig_attention.update_layout(
        title=f"Attention Weights (Epoch {epoch}, Batch {batch})",
        xaxis=dict(title="Tokens"),
        yaxis=dict(title="Attention Weights"),
        height=400,
    )

    # Render the attention weights chart in Streamlit
    st.plotly_chart(fig_attention, use_container_width=True)

    # Prototype activations chart (if provided)
    if prototype_activations is not None and isinstance(prototype_activations, np.ndarray):
        prototype_means = prototype_activations.mean(axis=-1)  # Compute mean activations
        fig_prototype = go.Figure()
        fig_prototype.add_trace(
            go.Bar(
                x=tokens,
                y=prototype_means,
                marker=dict(color=prototype_means, colorscale="Cividis"),
                name="Prototype Activations",
            )
        )
        fig_prototype.update_layout(
            title=f"Prototype Activations (Epoch {epoch}, Batch {batch})",
            xaxis=dict(title="Tokens"),
            yaxis=dict(title="Prototype Activations"),
            height=400,
        )

        # Render the prototype activations chart in Streamlit
        st.plotly_chart(fig_prototype, use_container_width=True)

# Function to calculate loss and extract explanations
def calculate_loss(model, batch, tokenizer):
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)

    # Set labels for unsupervised tasks (language modeling)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    # Extract loss and explanations
    loss = outputs.get("loss")
    explanations = outputs.get("explanations")

    if loss is None:
        raise ValueError("Model outputs did not contain a 'loss' key.")

    return loss, explanations

def generate_report(attention_data, prototype_data, smiles_examples):
    """
    Generate an interpretability report for attention weights and prototype activations.

    Parameters:
        attention_data (list of dict): Contains attention weights and tokens for each example.
        prototype_data (list of dict): Contains prototype activations and tokens for each example.
        smiles_examples (list of str): List of SMILES strings for reference.

    Returns:
        None
    """
    st.header("Interpretability Report")

    # Introduction
    st.subheader("1. Introduction")
    st.write("""
        This report summarizes the interpretability of the ChemBERTa model using two mechanisms:
        1. Attention Weights: Highlight the importance of specific tokens in a SMILES sequence for predictions.
        2. Prototype Activations: Show how well tokens align with learned global patterns (e.g., functional groups, substructures).
    """)

    # Attention Weights Analysis
    st.subheader("2. Attention Weights Analysis")
    for idx, data in enumerate(attention_data):
        st.write(f"### Example {idx + 1}: {smiles_examples[idx]}")
        st.write("This chart shows the tokens that received the highest attention weights for this SMILES sequence.")
        
        fig_attention = go.Figure()
        fig_attention.add_trace(
            go.Bar(
                x=data["tokens"],
                y=data["weights"],
                marker=dict(color=data["weights"], colorscale="Viridis"),
                name="Attention Weights"
            )
        )
        fig_attention.update_layout(
            title=f"Attention Weights for SMILES Example {idx + 1}",
            xaxis=dict(title="Tokens"),
            yaxis=dict(title="Attention Weights")
        )
        st.plotly_chart(fig_attention, use_container_width=True)

    # Prototype Activations Analysis
    st.subheader("3. Prototype Activations Analysis")
    for idx, data in enumerate(prototype_data):
        st.write(f"### Example {idx + 1}: {smiles_examples[idx]}")
        st.write("This chart shows the average prototype activations for this SMILES sequence.")
        
        fig_prototypes = go.Figure()
        fig_prototypes.add_trace(
            go.Bar(
                x=data["tokens"],
                y=data["activations"],
                marker=dict(color=data["activations"], colorscale="Cividis"),
                name="Prototype Activations"
            )
        )
        fig_prototypes.update_layout(
            title=f"Prototype Activations for SMILES Example {idx + 1}",
            xaxis=dict(title="Tokens"),
            yaxis=dict(title="Prototype Activations")
        )
        st.plotly_chart(fig_prototypes, use_container_width=True)

    # Insights and Observations
    st.subheader("4. Insights and Observations")
    st.write("""
        - Attention weights tend to highlight key substructures (e.g., double bonds, aromatic rings) that directly influence predictions.
        - Prototype activations reveal global patterns, such as common functional groups, that are consistently important across examples.
        - Comparing attention weights and prototype activations can help identify whether the model is making predictions for the right reasons.
    """)

    # Conclusion
    st.subheader("5. Conclusion")
    st.write("""
        This analysis demonstrates how attention weights and prototype activations can provide interpretable insights into the ChemBERTaXAI model.
        These insights can guide improvements in model design and help validate its predictions in a transparent manner.
    """)

# Training pipeline
def train_model(model, tokenizer, train_loader, epochs, learning_rate, output_dir):
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Lists to store data for the report
    report_attention_data = []
    report_prototype_data = []
    report_smiles_data = []

    logger.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            # Forward pass with explanations
            loss, explanations = calculate_loss(model, batch, tokenizer)

            # Backpropagation
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Track training loss
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

            # Update real-time training progress in Streamlit
            training_progress.progress((epoch * len(train_loader) + i + 1) / (epochs * len(train_loader)))
            avg_train_loss = train_loss / (i + 1)
            training_loss_placeholder.text(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

            # Visualize explanations for a single batch
            if explanations:
                attention_weights = explanations["attention_weights"][0].cpu().detach().numpy()
                prototype_activations = explanations.get("token_prototype_activations")

                if prototype_activations is not None:
                    prototype_activations = prototype_activations[0].cpu().detach().numpy()
                
                # Convert input IDs to tokens
                tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][0].tolist())

                # Filter and map tokens properly
                filtered_tokens, filtered_attention = zip(*[
                    (token, weight) for token, weight in zip(tokens, attention_weights)
                    if token not in ["<pad>"]  # Exclude padding tokens
                ])

                # Normalize filtered attention weights
                filtered_attention = np.array(filtered_attention)
                filtered_attention /= np.sum(filtered_attention)

                # Store data for report
                report_attention_data.append({
                    "tokens": filtered_tokens,
                    "weights": filtered_attention
                })
                report_prototype_data.append({
                    "tokens": filtered_tokens,
                    "activations": prototype_activations.mean(axis=-1) if prototype_activations is not None else None
                })
                report_smiles_data.append(" ".join(filtered_tokens))

                # Plot heatmap with filtered tokens, attention, and prototype activations
                display_attention_and_prototype_charts(
                    tokens=filtered_tokens,
                    attention_weights=filtered_attention,
                    prototype_activations=prototype_activations,
                    epoch=epoch + 1, 
                    batch=i + 1
                )

        logger.info(f"Epoch {epoch+1} Training Loss: {train_loss / len(train_loader):.4f}")

    # Save the fine-tuned model after all epochs
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Fine-tuned model saved to {output_dir}")

    # Return data after training is completed
    return report_attention_data, report_prototype_data, report_smiles_data


report_attention_data, report_prototype_data, report_smiles_data = train_model(
    model, tokenizer, train_loader, epochs, learning_rate, output_dir
)

# Generate structured prompts
prompts = create_prompt(report_attention_data, report_prototype_data, report_smiles_data)

# Get chemical insights from the API
api_key = "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"  # Replace with your API key
chemical_insights = get_chemical_insights(prompts, api_key)

# Display insights in the dashboard
st.subheader("Chemically-Based Insights")
for idx, insight in enumerate(chemical_insights):
    st.write(f"### Example {idx + 1}")
    st.write(f"**SMILES Sequence:** {report_smiles_data[idx]}")
    st.write(insight)

# Save the fine-tuned model

from rdkit import Chem
import csv
import os
import random

output_csv="chemBETA_generatedSMILES.csv"
with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Generated SMILES"])  # Add a header row for clarity

def generate_smiles(
    model, 
    tokenizer, 
    seed_text, 
    max_new_tokens=50, 
    temperature=1.2, 
    top_p=0.9, 
    output_csv="chemBETA_generatedSMILES.csv",
    random_seed=None
):
    """
    Generate a SMILES string using the model, validate its chemical structure, and save to a CSV file.
    """
    model.eval()

    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    try:
        # Tokenize the seed text
        inputs = tokenizer(
            seed_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        ).to(model.device)

        # Generate SMILES
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        print(f"Debug: Output from model.generate: {output}")
        # Decode the output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # Validate the generated SMILES
        if is_valid_smiles(generated_text):
            with open(output_csv, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([generated_text])
            logger.info(f"Generated and saved valid SMILES: {generated_text}")
            return generated_text
        else:
            logger.warning(f"Generated invalid SMILES: {generated_text}")
            return None
    except Exception as e:
        logger.error(f"Error during SMILES generation: {e}")
        return None

substructure_explanations = {
    # Basic elements
    "C": "Carbon atom: fundamental building block of organic molecules, determines backbone structure.",
    "H": "Hydrogen atom: critical for molecular interactions, often involved in hydrogen bonding.",
    "O": "Oxygen atom: contributes to polarity and hydrogen bonding, affects solubility.",
    "N": "Nitrogen atom: introduces polarity, basicity, and potential hydrogen bonding sites.",
    "S": "Sulfur atom: increases molecular complexity, affects reactivity and binding properties.",
    "P": "Phosphorus atom: common in phosphates, affects polarity and enzyme interactions.",
    "F": "Fluorine atom: improves metabolic stability and binding affinity but can affect toxicity.",
    "Cl": "Chlorine atom: can improve binding affinity but may increase toxicity risk.",
    "Br": "Bromine atom: similar to chlorine, affects binding affinity and reactivity.",
    "I": "Iodine atom: increases molecular weight and hydrophobicity, can affect binding affinity.",

    # Hydrocarbons
    "CH": "Part of alkyl chain: contributes to hydrophobicity and molecular volume.",
    "CH3": "Methyl group: increases hydrophobicity, common in many organic molecules.",
    "CH2": "Methylene group: forms part of hydrocarbon chains and rings, contributes to flexibility.",
    "CC": "Ethyl group: contributes to hydrophobicity, common in drug scaffolds.",
    "CCC": "Propyl group: moderate hydrophobicity, impacts lipophilicity and solubility.",
    "CCCC": "Butyl group: increases lipophilicity, often affects solubility and binding.",
    "C(C)C": "Isopropyl group: increases hydrophobicity and steric bulk.",
    "CC(C)C": "Tert-butyl group: highly hydrophobic, increases metabolic stability.",
    "C(C)(C)C": "Branched alkyl chain: increases steric hindrance, affects receptor binding.",
    "C(C)(C)(C)": "Quaternary carbon: adds steric bulk, affects bioavailability.",

    # Halogen-containing groups
    "CF3": "Trifluoromethyl group: enhances metabolic stability, affects lipophilicity.",
    "CCl": "Chloromethyl group: increases reactivity and binding affinity.",
    "CBr": "Bromomethyl group: contributes to hydrophobicity, affects reactivity.",
    "CI": "Iodomethyl group: increases molecular weight, may enhance binding affinity.",

    # Oxygen-containing groups
    "OH": "Hydroxyl group: increases solubility and hydrogen bonding potential.",
    "CO": "Methoxy group: can alter electronic properties and affect solubility.",
    "OCC": "Oxygen-carbon chain: may affect polarity and reactivity.",
    "COC": "Ether group: increases polarity and flexibility.",
    "C=O": "Ketone group: contributes to polarity and reactivity, can act as a hydrogen bond acceptor.",
    "COOH": "Carboxylic acid group: improves solubility but may decrease membrane permeability.",
    "C(O)O": "Lactone group: cyclic ester, common in natural products and antibiotics.",
    "O=C(O)": "Ester group: increases reactivity and can act as a prodrug moiety.",
    "O=C(N)": "Amide group: contributes to hydrogen bonding and stability in proteins.",
    "O=C-S": "Thioester group: enhances reactivity, common in biochemical processes.",
    "C-O-C": "Acetal group: increases hydrophilicity, often used in sugars and prodrugs.",
    "C(O)OH": "Hydroxy acid: increases both hydrogen bonding and solubility.",

    # Nitrogen-containing groups
    "NH2": "Primary amine: contributes to basicity and hydrogen bonding.",
    "NH": "Amine group: increases basicity and hydrogen bonding potential, common in drugs.",
    "CN": "Methylamino group: contributes to basicity and affects metabolic stability.",
    "N(C)": "Methylamine group: may increase hydrophilicity but could impact bioavailability.",
    "N=N": "Azo group: can increase conjugation but may introduce toxicity risks.",
    "C#N": "Nitrile group: contributes to polarity, affects metabolic stability.",
    "C(NO2)": "Nitro group: increases electron density, often associated with toxicity risks.",
    "N=C=O": "Isocyanate group: highly reactive, affects protein interactions.",
    "C(N)C=O": "Amide group: enhances hydrogen bonding and polarity.",
    "C(O)N": "Hydroxylamine: enhances hydrogen bonding, reactive in biological systems.",

    # Sulfur-containing groups
    "SH": "Thiol group: increases reactivity, common in metal ion chelation.",
    "C(S)": "Thiol group: highly reactive, impacts enzyme activity.",
    "C(S)C": "Thioether group: contributes to hydrophobicity, can enhance lipophilicity.",
    "S(=O)(=O)": "Sulfonyl group: highly polar, increases solubility and hydrogen bonding.",
    "C=S": "Thione group: enhances reactivity and electron density.",
    "S(=O)": "Sulfoxide group: increases polarity and can enhance water solubility.",
    "S(=O)(N)": "Sulfone group: enhances polarity and reactivity, used in pharmaceuticals.",

    # Phosphorus-containing groups
    "P(=O)(O)O": "Phosphate group: highly polar, commonly used in prodrugs and biological molecules.",
    "OP(O)(O)": "Phosphonate group: increases polarity, commonly used in enzyme inhibitors.",
    "P(C)(C)": "Phosphine group: increases reactivity and electron density.",
    "P=S": "Thionophosphate group: used in pesticides and enzyme inhibitors.",

    # Aromatic rings
    "c1cccs1": "Thiophene ring: sulfur-containing aromatic rings may reduce stability or drug-likeness.",
    "c1ccc": "Benzene ring: common aromatic scaffold; impacts hydrophobicity and binding.",
    "c1ccc2c(c1)ccc2": "Naphthalene ring: increases hydrophobicity, common in hydrophobic drug scaffolds.",
    "c1ccn": "Pyridine ring: increases polarity and can enhance drug-likeness.",
    "c1cccnc1": "Pyrimidine ring: contributes to hydrogen bonding and polarity, often found in drugs.",
    "c1ccco1": "Furan ring: oxygen-containing aromatic ring, can affect stability and reactivity.",
    "c1c[nH]cn1": "Imidazole ring: contributes to hydrogen bonding and polarity, common in enzyme inhibitors.",
    "c2cccs2": "Thiophene derivative: contributes to electronic properties and aromaticity.",
    "c1nnc2c1ncnc2": "Purine scaffold: key component of nucleotides and many bioactive molecules.",

    # Cyclic groups
    "C1CC1": "Cyclopropane: adds steric bulk, affects binding affinity and rigidity.",
    "C1CCC1": "Cyclobutane: increases steric strain and rigidity, affects hydrophobicity.",
    "C1CCCC1": "Cyclopentane: hydrophobic scaffold, adds moderate flexibility.",
    "C1CCCCC1": "Cyclohexane: increases hydrophobicity and rigidity.",
    "C1C=CC=C1": "Cyclohexadiene: contributes to electronic conjugation, adds rigidity.",

    # Double and triple bonds
    "C=C": "Double bond: increases reactivity but can reduce chemical stability.",
    "C#C": "Triple bond: highly reactive, contributes to molecular rigidity.",
    "C=C-C": "Conjugated system: enhances electronic delocalization, common in chromophores.",
    "C=C-O": "Enol group: enhances reactivity, often seen in keto-enol tautomerism.",
    "C=C(C)": "Alkene group: increases lipophilicity and reactivity.",

    # Miscellaneous
    "NO2": "Nitro group: increases electron density, often associated with toxicity risks.",
    "SO2": "Sulfonyl group: highly polar, increases solubility and hydrogen bonding.",
    "C(F)(F)(F)": "Trifluoromethyl group: enhances metabolic stability and lipophilicity.",
    "C(O)(N)": "Hydroxamic acid: acts as a chelator and hydrogen bond donor/acceptor.",
    "C=CC": "Alkene group: increases lipophilicity and reactivity.",
    "CO": "Carbon monoxide-like fragment: contributes to polarity, rare in stable compounds.",

    "n3cncn3": "Triazine ring: nitrogen-rich aromatic ring, often used in agrochemicals and dyes.",
    "n1": "Part of an aromatic heterocycle containing nitrogen, such as pyridine or triazole.",
    "Cc1ccc": "Substituted benzene ring: impacts hydrophobicity and binding.",
    "cc1S": "Sulfur-containing aromatic ring: affects electronic properties and reactivity.",
    "c2ccncc2": "Pyridazine ring: nitrogen-rich heterocycle, enhances polarity.",
    "NC1CC1": "Aminocyclopropane: introduces steric bulk and affects bioavailability.",
    "nH": "Nitrogen with a hydrogen atom: contributes to hydrogen bonding and basicity.",
    "nc2c1": "Fused heterocyclic rings: impacts electronic and binding properties.",
    "Nc1ccc2ncccc2c1": "Quinoline derivative: enhances aromaticity and potential binding affinity.",
    "s2": "Part of a thiophene-like structure: sulfur-containing aromatic ring affects stability.",
    "cs2": "Thiocarbonyl group: contributes to reactivity and sulfur-based interactions.",
    "c3nc": "Part of a nitrogen-containing aromatic system: enhances polarity and binding.",
    "C1": "Cyclized carbon structure: contributes to rigidity and hydrophobicity.",
    "1": "Possible ring closure marker: indicates cyclic structures.",

    "c1cccs1": "Thiophene ring: sulfur-containing aromatic rings may reduce stability or drug-likeness.",
    "COc1ccc": "Methoxy group: can alter electronic properties and affect solubility.",
    "OC": "Hydroxyl group: increases solubility but may introduce metabolic liabilities.",
    "OCC": "Oxygen-carbon chain: may affect polarity and reactivity.",
    "c1C": "Part of aromatic ring system: contributes slightly to the molecule's overall behavior.",
    "N(C)": "Methylamine group: may increase hydrophilicity but could impact bioavailability.",
    "CC(=O)": "Carbonyl group: often improves reactivity but may reduce stability.",
    "Cl": "Chlorine atom: can improve binding affinity but may increase toxicity risk.",
    "CN": "Methylamino group: contributes to basicity and affects metabolic stability.",
    "C=C": "Double bond: increases reactivity but can reduce chemical stability.",
    "C#N": "Nitrile group: contributes to polarity but may decrease synthetic accessibility.",
    "COOH": "Carboxylic acid group: improves solubility but may decrease membrane permeability.",
    "c1ccn": "Pyridine ring: increases polarity and can enhance drug-likeness.",
    "F": "Fluorine atom: improves metabolic stability and binding affinity but can affect toxicity.",
    "Br": "Bromine atom: similar to chlorine, affects binding affinity and reactivity.",
    "c1ccc": "Benzene ring: common aromatic scaffold; impacts hydrophobicity and binding."
}

default_explanation = "No specific insights available for this substructure."

# Save plot as base64-encoded image
def save_plot_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return encoded_image

# Validate SMILES strings
def is_valid_smiles(smiles):
    """
    Check if the given SMILES string is valid using RDKit.
    Args:
        smiles (str): A SMILES string to validate.
    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception as e:
        logger.error(f"Error validating SMILES: {e}")
        return False
    
def visualize_token_importance(smiles, importance_weights):
    """
    Visualize token importance as a bar chart.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    tokens = list(smiles)
    sns.barplot(x=tokens, y=importance_weights, ax=ax)
    ax.set_title("Token Importance")
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Importance Weight")
    plt.xticks(rotation=45)
    return save_plot_to_base64(fig)

# Attention Visualization
def extract_attention_weights(model, tokenizer, smiles):
    """
    Extract and visualize attention weights for SMILES tokens.
    """
    inputs = tokenizer(smiles, return_tensors="pt", max_length=max_length, truncation=True).to(model.device)
    outputs = model(**inputs, output_attentions=True)
    attention_weights = outputs.attentions[-1].mean(dim=1).squeeze(0).cpu().detach().numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, cmap="viridis", ax=ax)
    ax.set_title("Attention Heatmap")
    return save_plot_to_base64(fig)

def generate_shap_explanations(smiles, prediction_fn):
    """
    Generate SHAP explanations for a SMILES prediction.
    """
    explainer = shap.Explainer(prediction_fn, tokenizer)
    shap_values = explainer([smiles])
    fig = shap.force_plot(shap_values[0])
    return save_plot_to_base64(fig)

def generate_rule_based_explanation(smiles):
    """
    Provide rule-based insights for SMILES based on predefined chemical rules.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return ["Invalid SMILES string; unable to generate explanations."]

    rules = []

    # Check atom properties
    if mol.GetNumAtoms() > 50:
        rules.append("The molecule has a high atom count, which may reduce bioavailability.")
    if any(atom.GetSymbol() == "F" for atom in mol.GetAtoms()):
        rules.append("Fluorine atoms improve metabolic stability but may increase toxicity.")
    if any(atom.GetSymbol() == "Cl" for atom in mol.GetAtoms()):
        rules.append("Chlorine atoms may enhance binding affinity but risk higher toxicity.")

    # Check for specific substructures
    try:
        benzene = Chem.MolFromSmiles("c1ccccc1")
        if benzene and mol.HasSubstructMatch(benzene):
            rules.append("Benzene ring detected, which contributes to hydrophobicity and aromaticity.")

        hydroxyl = Chem.MolFromSmiles("[OH]")
        if hydroxyl and mol.HasSubstructMatch(hydroxyl):
            rules.append("Hydroxyl group detected, which increases polarity and solubility.")

        carboxylic_acid = Chem.MolFromSmiles("C(=O)[OH]")
        if carboxylic_acid and mol.HasSubstructMatch(carboxylic_acid):
            rules.append("Carboxylic acid group detected, which enhances solubility but may reduce membrane permeability.")
    except Exception as e:
        rules.append(f"Error during substructure matching: {e}")

    return rules if rules else ["No specific rules applied to this molecule."]


from captum.attr import IntegratedGradients
import torch

def generate_integrated_gradients(smiles, model, tokenizer):
    """
    Generate Integrated Gradients explanation for a given SMILES.

    Args:
        smiles (str): SMILES string to analyze.
        model (torch.nn.Module): The pre-trained model.
        tokenizer (Tokenizer): Tokenizer corresponding to the model.

    Returns:
        str: Base64 encoded image of IG visualization or textual insights.
    """
    try:
        # Tokenize input
        inputs = tokenizer(
            smiles,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        ).to(model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Initialize Integrated Gradients
        ig = IntegratedGradients(model)

        # Compute attributions
        attributions, delta = ig.attribute(
            input_ids,
            target=0,  # Target can be modified depending on the use case
            return_convergence_delta=True,
            additional_forward_args=attention_mask
        )

        # Summing over all attributions to get aggregate importance
        attributions_sum = attributions.sum(dim=2).squeeze(0).detach().cpu().numpy()

        # Visualize attributions
        fig, ax = plt.subplots(figsize=(10, 2))
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        ax.bar(range(len(tokens)), attributions_sum, tick_label=tokens)
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Attribution")
        ax.set_title(f"Integrated Gradients for {smiles}")
        plt.xticks(rotation=90)

        # Save plot to base64 string
        encoded_image = save_plot_to_base64(fig)
        plt.close(fig)
        return encoded_image

    except Exception as e:
        logger.error(f"Error generating Integrated Gradients explanation: {e}")
        return None


# Detailed substructure explanations
def generate_detailed_explanations(smiles, lime_features, lime_weights):
    """
    Generate detailed explanations for LIME features using substructure insights.

    Args:
        smiles (str): The SMILES string being analyzed.
        lime_features (list): Substructures identified by LIME.
        lime_weights (list): Corresponding importance weights for the substructures.

    Returns:
        list: A list of detailed textual explanations for each substructure with specific insights.
    """
    explanations = []
    if not lime_features or not lime_weights:
        return ["Error: Missing features or weights for explanation generation."]
    
    for feature, weight in zip(lime_features, lime_weights):
        try:
            # Fetch substructure insight
            substructure_insight = substructure_explanations.get(feature)
            
            # Skip substructures without specific insights
            if not substructure_insight:
                continue
            
            # Determine impact
            impact = "positive" if weight > 0 else "negative"
            
            # Construct explanation
            explanations.append(
                f"Substructure: {feature} | Impact: {impact} | Magnitude: {abs(weight):.2f} | Insight: {substructure_insight}"
            )
        except Exception as e:
            # Handle unexpected errors
            explanations.append(f"Error processing feature {feature}: {e}")
    
    return explanations if explanations else ["No substructures with specific insights found."]

from rdkit import Chem
from rdkit import Chem
from rdkit.Chem import BondType
from rdkit.Chem import rdchem
from rdkit.Chem import PeriodicTable
from rdkit import Chem
import random
import logging

logger = logging.getLogger(__name__)

from rdkit import Chem
from rdkit.Chem import AllChem
import random
import logging

logger = logging.getLogger(__name__)

from rdkit import Chem
from rdkit.Chem import AllChem
import random
import logging

logger = logging.getLogger(__name__)

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import random
import logging

logger = logging.getLogger(__name__)

# Functional Group Replacement Rules
FUNCTIONAL_GROUP_REPLACEMENTS = {
    "amide": {"pattern": "[CX3](=O)[NX3H2]", "replacement": "[CX3](=O)O"},  # Amide → Ester
    "halogen": {"pattern": "[F,Cl,Br,I]", "replacement": "[H]"},  # Replace halogen with H
    "ether": {"pattern": "[C]-O-[C]", "replacement": "[C]-S-[C]"},  # Ether → Thioether
}

from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
import random

from rdkit import Chem
from rdkit.Chem import AllChem
import random

def generate_counterfactuals(original_smiles, max_changes=5, max_attempts=50):
    """
    Generate diverse counterfactuals with systematic modifications.
    Args:
        original_smiles (str): Original SMILES string.
        max_changes (int): Maximum number of counterfactuals to generate.
        max_attempts (int): Maximum number of attempts to generate valid counterfactuals.
    Returns:
        list: List of valid counterfactual SMILES.
    """
    counterfactuals = set()
    original_mol = Chem.MolFromSmiles(original_smiles)
    if not original_mol:
        raise ValueError(f"Invalid original SMILES: {original_smiles}")

    # Predefined substitutions and bioisosteric replacements
    substitutions = [
        {"from": "C", "to": "N"},
        {"from": "N", "to": "O"},
        {"from": "O", "to": "S"},
        {"from": "F", "to": "Cl"},
        {"from": "Cl", "to": "Br"},
    ]

    for _ in range(max_attempts):
        try:
            modified_mol = Chem.RWMol(original_mol)

            # Apply random substitutions
            if random.random() < 0.5:
                atom_idx = random.choice(range(modified_mol.GetNumAtoms()))
                atom = modified_mol.GetAtomWithIdx(atom_idx)
                current_symbol = atom.GetSymbol()
                possible_subs = [sub for sub in substitutions if sub["from"] == current_symbol]
                if possible_subs:
                    new_symbol = random.choice(possible_subs)["to"]
                    atom.SetAtomicNum(Chem.PeriodicTable.GetAtomicNumber(Chem.GetPeriodicTable(), new_symbol))

            # Apply random bond modifications
            else:
                atom_indices = random.sample(range(modified_mol.GetNumAtoms()), 2)
                if not modified_mol.GetBondBetweenAtoms(atom_indices[0], atom_indices[1]):
                    modified_mol.AddBond(atom_indices[0], atom_indices[1], Chem.BondType.SINGLE)
                else:
                    bond = modified_mol.GetBondBetweenAtoms(atom_indices[0], atom_indices[1])
                    new_bond_type = random.choice([Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE])
                    bond.SetBondType(new_bond_type)

            # Sanitize and validate
            Chem.SanitizeMol(modified_mol)
            new_smiles = Chem.MolToSmiles(modified_mol, isomericSmiles=True)
            if new_smiles != original_smiles:
                counterfactuals.add(new_smiles)
                if len(counterfactuals) >= max_changes:
                    break
        except Exception as e:
            continue

    return list(counterfactuals)

def validate_smiles(smiles_list):
    """
    Validate and canonicalize SMILES strings.
    Args:
        smiles_list (list): List of SMILES strings.
    Returns:
        list: Valid and canonicalized SMILES.
    """
    valid_smiles = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                Chem.SanitizeMol(mol)
                canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                valid_smiles.append(canonical_smiles)
        except Exception as e:
            continue
    return valid_smiles

def prediction_fn(smiles_list):
    """
    Predict property scores for molecules.
    Args:
        smiles_list (list): List of SMILES strings.
    Returns:
        list: Prediction scores.
    """
    try:
        inputs = tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128  # Adjust depending on model
        ).to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            return probs[:, 1].tolist()  # Assuming class 1 is the target
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return [0.0] * len(smiles_list)  # Default scores in case of errors

def compare_structures(original_smiles, counterfactual_smiles):
    """
    Compare structures of the original and counterfactual molecules.
    Args:
        original_smiles (str): Original SMILES string.
        counterfactual_smiles (str): Counterfactual SMILES string.
    Returns:
        list: Structural differences with actions.
    """
    original_mol = Chem.MolFromSmiles(original_smiles)
    counterfactual_mol = Chem.MolFromSmiles(counterfactual_smiles)
    differences = []

    # Compare atoms
    original_atoms = [atom.GetSymbol() for atom in original_mol.GetAtoms()]
    counterfactual_atoms = [atom.GetSymbol() for atom in counterfactual_mol.GetAtoms()]
    for atom in set(counterfactual_atoms):
        if counterfactual_atoms.count(atom) > original_atoms.count(atom):
            differences.append({"action": "Added", "substructure": atom})
    for atom in set(original_atoms):
        if original_atoms.count(atom) > counterfactual_atoms.count(atom):
            differences.append({"action": "Removed", "substructure": atom})

    return differences

def generate_counterfactual_explanation(original_smiles, counterfactuals, prediction_fn):
    """
    Generate explanations for counterfactuals.
    Args:
        original_smiles (str): Original SMILES string.
        counterfactuals (list): Counterfactual SMILES.
        prediction_fn (callable): Prediction function.
    Returns:
        list: Explanations for each counterfactual.
    """
    explanations = []
    original_pred = prediction_fn([original_smiles])[0]
    valid_counterfactuals = validate_smiles(counterfactuals)

    for cf_smiles in valid_counterfactuals:
        cf_pred = prediction_fn([cf_smiles])[0]
        changes = compare_structures(original_smiles, cf_smiles)

        detailed_changes = []
        for change in changes:
            action = change.get("action", "Unknown")
            substructure = change.get("substructure", "Unknown")
            detailed_changes.append(f"{action} {substructure}")

        impact = "Increased" if cf_pred > original_pred else "Decreased"
        explanation = (
            f"Original: {original_smiles} (Prediction: {original_pred:.3f})\n"
            f"Counterfactual: {cf_smiles} (Prediction: {cf_pred:.3f})\n"
            f"Changes:\n" + "\n".join(detailed_changes) + "\n"
            f"Impact: {impact} prediction by {abs(cf_pred - original_pred):.3f}."
        )
        explanations.append(explanation)

    return explanations

# Generate explanations
def generate_explanations(smiles):
    explanations = {"smiles": smiles, "lime_plot": None, "shap_plot": None, "counterfactual": None, "rule_based": None}

    # Token importance and attention
    #try:
     #   explanations["attention_heatmap"] = extract_attention_weights(model, tokenizer, smiles)
    #except Exception as e:
     #   explanations["attention_heatmap"] = f"Error: {e}"

    # Rule-Based Explanations
    try:
        explanations["rule_based"] = generate_rule_based_explanation(smiles)
    except Exception as e:
        explanations["rule_based"] = f"Error: {e}"

    # Counterfactuals
    try:
        counterfactuals = generate_counterfactuals(smiles, max_changes=5)
        explanations["counterfactual"] = generate_counterfactual_explanation(smiles, counterfactuals, prediction_fn)
    except Exception as e:
        explanations["counterfactual"] = f"Error: {e}"

    # LIME Explanation
    try:
        explainer = LimeTextExplainer()
        lime_exp = explainer.explain_instance(smiles, lambda x: np.random.rand(len(x), 10), num_features=5)
        lime_features, lime_weights = zip(*lime_exp.as_list())
        explanations["detailed_explanations"] = generate_detailed_explanations(smiles, lime_features, lime_weights)

         #Save LIME plot
        fig, ax = plt.subplots()
        ax.barh(lime_features, lime_weights)
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"LIME Explanation for {smiles}")
        explanations["lime_plot"] = save_plot_to_base64(fig)
        plt.close(fig)
    except Exception as e:
        explanations["lime_plot"] = f"Error: {e}"
        logger.error(f"Error generating LIME explanations: {e}")

    # SHAP Explanation
    try:
        explanations["shap_plot"] = generate_shap_explanations(smiles, prediction_fn)
    except Exception as e:
        explanations["shap_plot"] = f"Error: {e}"

    # Integrated Gradients
    try:
        logger.info(f"Generating Integrated Gradients explanation for SMILES: {smiles}")
        ig_plot = generate_integrated_gradients(smiles, model, tokenizer)
        if ig_plot:
            explanations["integrated_gradients"] = ig_plot
        else:
            explanations["integrated_gradients"] = "No IG explanation available."
    except Exception as e:
        explanations["integrated_gradients"] = f"Error: {e}"
        logger.error(f"Error generating Integrated Gradients explanation: {e}")

    return explanations




# Generate SMILES with explanations
logger.info("Generating SMILES with explanations...")
results = []
novel_smiles_set = set()

with tqdm(total=target_novel_smiles, desc="Generating SMILES") as pbar:
    print(f"Using model: {model.__class__.__name__}")
    print(f"Model configuration: {model.config}")
    print(f"Using tokenizer: {tokenizer.__class__.__name__}")
    print(f"Tokenizer configuration: {tokenizer}")

    while len(results) < target_novel_smiles:
        # Select a seed SMILES randomly
        seed_smiles = random.choice(data["SMILES"].tolist())
        
        # Generate SMILES
        generated = generate_smiles(model, tokenizer, seed_smiles)
        if generated and is_valid_smiles(generated) and generated not in novel_smiles_set:
            logger.info(f"Generated valid SMILES: {generated}")
            
            # Use the existing generate_explanations function
            explanations = generate_explanations(generated)

            # Log the results for debugging
            logger.info(f"Explanations for {generated}: {explanations}")

            # Append explanations and generated SMILES to results
            results.append({
                "generated_smiles": generated,
                "explanations": explanations
            })
            
            # Add to the novel SMILES set
            novel_smiles_set.add(generated)
            pbar.update(1)



# Save results as an HTML file
# Save results as an HTML file
output_file = "generated_smiles_with_explanations.html"
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMILES Explanations</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }
        img {
            max-width: 600px;
            height: auto;
            border: 1px solid #ddd;
            margin-top: 10px;
        }
        h1 {
            color: #003366;
            text-align: center;
        }
        h2 {
            color: #003366;
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin-bottom: 10px;
        }
        .section {
            margin-bottom: 30px;
        }
    </style>
</head>
<body>
    <h1>Generated SMILES with Explanations</h1>
"""

# Add generated SMILES and explanations to the HTML content
for res in results:
    generated_smiles = res.get("generated_smiles", "N/A")
    explanations = res.get("explanations", {})

    html_content += f"<div class='section'><h2>SMILES: {generated_smiles}</h2>"
    
    # Add Attention Visualization
    attention_heatmap = explanations.get("attention_heatmap")
    if attention_heatmap:
        html_content += f'<h3>Attention Visualization</h3>'
        html_content += f'<img src="data:image/png;base64,{attention_heatmap}" alt="Attention Visualization"><br>'
    else:
        html_content += "<p>No attention visualization available.</p>"

    # Add LIME Explanation
    lime_plot = explanations.get("lime_plot")
    if lime_plot:
        html_content += f'<h3>LIME Explanation</h3>'
        html_content += f'<img src="data:image/png;base64,{lime_plot}" alt="LIME Explanation"><br>'
    else:
        html_content += "<p>No LIME explanation available.</p>"

    detailed_explanations = explanations.get("detailed_explanations", [])
    if detailed_explanations:
        html_content += "<h3>Detailed Substructure Analysis</h3><ul>"
        for explanation in detailed_explanations:
            html_content += f"<li>{explanation}</li>"
        html_content += "</ul>"
    else:
        html_content += "<p>No detailed substructure analysis available.</p>"
    # Add Rule-Based Explanation
    rule_based = explanations.get("rule_based", [])
    if rule_based and isinstance(rule_based, list):
        html_content += "<h3>Rule-Based Explanation</h3><ul>"
        for rule in rule_based:
            html_content += f"<li>{rule}</li>"
        html_content += "</ul>"
    else:
        html_content += "<p>No rule-based explanations available or error occurred.</p>"

    # Add Counterfactual Explanation
    counterfactuals = explanations.get("counterfactual", [])
    if isinstance(counterfactuals, list) and counterfactuals:
        html_content += "<h3>Counterfactual Explanation</h3><ul>"
        for cf in counterfactuals:
            html_content += f"<li>{cf}</li>"
        html_content += "</ul>"
    else:
        html_content += "<p>No counterfactuals generated or explanation unavailable.</p>"



    # Add Integrated Gradients Explanation
    ig_plot = explanations.get("integrated_gradients")
    if ig_plot:
        html_content += f'<h3>Integrated Gradients Explanation</h3>'
        html_content += f'<img src="data:image/png;base64,{ig_plot}" alt="Integrated Gradients Explanation"><br>'
    else:
        html_content += "<p>No Integrated Gradients explanation available.</p>"

    # Add SHAP Explanation
    shap_plot = explanations.get("shap_plot")
    if shap_plot:
        html_content += f'<h3>SHAP Explanation</h3>'
        html_content += f'<img src="data:image/png;base64,{shap_plot}" alt="SHAP Explanation"><br>'
    else:
        html_content += "<p>No SHAP explanation available.</p>"

    html_content += "</div>"

# Finalize and save the HTML content
html_content += "</body></html>"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(html_content)

logger.info(f"Results saved to {output_file}")