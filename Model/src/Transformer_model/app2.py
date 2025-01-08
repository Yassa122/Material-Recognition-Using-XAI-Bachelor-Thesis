import math
import uuid
from flask import Flask, json, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from tqdm import tqdm  # Import the tqdm function
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
import torch
from torch.utils.data import Dataset, DataLoader
import time
import os
from torch import nn
from threading import Thread, Lock
from rdkit import Chem
from rdkit.Chem import Descriptors
import logging
import shap
import torch
import pandas as pd
from flask import jsonify, request
from tqdm import tqdm
from torch.utils.data import DataLoader
from rdkit import Chem
import os
import logging
import time
from threading import Thread
from io import BytesIO
from datetime import datetime
import numpy as np
import matplotlib
from flask import send_from_directory
from lime.lime_tabular import LimeTabularExplainer
import os
import openai
from flask import Flask, request, jsonify
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
)

# from dotenv import load_dotenv
import logging

matplotlib.use("Agg")  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt

# Flask App Initialization
app = Flask(__name__)
CORS(app)

# Global Variables
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model_lock = Lock()

# Logging setup
logging.basicConfig(level=logging.INFO)
# Status of the training
training_status = {"status": "idle", "message": ""}
prediction_status = {"status": "idle", "message": "", "progress": 0, "eta": None}


# Function to extract knowledge-based features
# Function to extract knowledge-based features
def extract_knowledge_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        molecular_weight = Descriptors.MolWt(mol)
        mol_logp = Descriptors.MolLogP(mol)  # Added MolLogP
        num_h_donors = Descriptors.NumHDonors(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        return [molecular_weight, mol_logp, num_h_donors, num_h_acceptors]
    else:
        return [0, 0, 0, 0]  # Updated to match the number of descriptors


def build_training_csv_with_descriptors(input_csv, output_csv):
    """
    Reads an input CSV of SMILES + property columns.
    Computes RDKit descriptors for each SMILES.
    Merges them into a new DataFrame with the property columns included.
    Saves to output_csv.
    """
    df = pd.read_csv(input_csv)
    if "SMILES" not in df.columns:
        raise ValueError("Input CSV must have a 'SMILES' column.")

    all_descriptors = []
    for sm in tqdm(df["SMILES"], desc="Computing RDKit descriptors"):
        desc_values = calculate_rdkit_descriptors(sm)
        all_descriptors.append(desc_values)
    desc_df = pd.DataFrame(all_descriptors, columns=descriptor_names)

    # Merge the descriptors and the original CSV
    merged_df = pd.concat([df.reset_index(drop=True), desc_df], axis=1)
    # e.g. columns: [SMILES, property1, property2, ... descriptor1, descriptor2, ...]

    # Save
    merged_df.to_csv(output_csv, index=False)
    return output_csv


def calculate_rdkit_descriptors(smiles):
    """
    Return a list of many RDKit descriptors.
    For consistency, you should define a global 'descriptor_functions'
    list so that the same ordering is used for train & predict.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        # Return a list of NaNs the same length as descriptor_functions
        return [float("nan")] * len(descriptor_functions)
    values = []
    for name, func in descriptor_functions:
        val = func(mol)
        values.append(val)
    return values


# We define a global list of descriptor functions from RDKit:
descriptor_functions = [
    (desc_name, desc_fn) for desc_name, desc_fn in Descriptors._descList
]
# If you want to exclude some descriptors (e.g. ones that break on certain molecules),
# you can remove them from this list.

# For easy reference
# Define RDKit descriptors used for training and prediction
descriptor_names = ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors"]


# Load SMILES Data
def load_smiles_data(csv_file, properties_present=True):
    data = pd.read_csv(csv_file)
    smiles = data["SMILES"].tolist()
    knowledge_features = [extract_knowledge_features(sm) for sm in smiles]

    if properties_present:
        # Drop 'mol' column if it exists
        targets = data.drop(columns=["SMILES", "mol"], errors="ignore")
        # Convert to numeric and handle missing values
        targets = targets.apply(pd.to_numeric, errors="coerce")
        valid_indices = targets.dropna().index
        # Log the number of valid and invalid rows
        logging.info(
            f"Total SMILES: {len(smiles)}, Valid SMILES with properties: {len(valid_indices)}"
        )
        smiles = [smiles[i] for i in valid_indices]
        knowledge_features = [knowledge_features[i] for i in valid_indices]
        targets = targets.loc[valid_indices].values
        return smiles, targets, knowledge_features
    else:
        return smiles, None, knowledge_features


# SMILESDataset Class
class SMILESDataset(Dataset):
    def __init__(
        self,
        smiles_list,
        knowledge_features,
        target_list=None,
        tokenizer=None,
        max_length=128,
    ):
        self.smiles_list = smiles_list
        self.knowledge_features = knowledge_features
        self.target_list = target_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        inputs = self.tokenizer(
            smiles,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "knowledge_features": torch.tensor(
                self.knowledge_features[idx], dtype=torch.float
            ),
        }
        if self.target_list is not None:
            item["labels"] = torch.tensor(self.target_list[idx], dtype=torch.float)
        return item


from rdkit.Chem import Descriptors

all_descriptor_funcs = [x for x in Descriptors._descList]


def compute_rdkit_descriptors_for_smiles(smiles: str):
    """
    Compute a dictionary of RDKit descriptors for a single SMILES string.
    """
    import numpy as np
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return NaN for all descriptors if SMILES is invalid
        return {func_name: np.nan for func_name, _ in all_descriptor_funcs} | {
            "Valid_SMILES": False
        }

    values = {}
    for func_name, func in all_descriptor_funcs:
        values[func_name] = func(mol)
    values["Valid_SMILES"] = True
    return values


def create_descriptor_csv(input_csv: str, output_csv: str):
    """
    Reads a CSV with at least a 'SMILES' column.
    For each row, computes only a subset of descriptors (MolWt, LogP, etc.)
    and merges them with the original columns. Saves to output_csv.
    """
    import pandas as pd

    df = pd.read_csv(input_csv)
    if "SMILES" not in df.columns:
        raise ValueError("CSV must contain a 'SMILES' column.")

    # Compute ONLY the relevant descriptors for each SMILES
    descriptor_dicts = df["SMILES"].apply(compute_receptor_binding_descriptors)
    desc_df = pd.DataFrame(list(descriptor_dicts))

    # Merge with the original DataFrame
    combined = pd.concat([df, desc_df], axis=1)

    # (Optional) Filter out invalid SMILES
    # combined = combined[combined["Valid_SMILES"] == True].copy()

    combined.to_csv(output_csv, index=False)
    print(f"Descriptor-augmented CSV saved to: {output_csv}")


class DescriptorDataset(Dataset):
    def __init__(
        self,
        df,
        target_columns=None,
        descriptor_columns=None,
        tokenizer=None,
        max_length=128,
    ):
        self.df = df.reset_index(drop=True)
        self.target_columns = target_columns or []
        self.descriptor_columns = descriptor_columns or []
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.smiles_list = df["SMILES"].tolist()
        self.descriptor_matrix = df[self.descriptor_columns].values.astype(float)

        if len(self.target_columns) > 0:
            self.target_matrix = df[self.target_columns].values.astype(float)
        else:
            self.target_matrix = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1) Tokenize SMILES
        smiles = self.smiles_list[idx]
        inputs = self.tokenizer(
            smiles,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        # 2) Descriptors
        descriptor_feats = torch.tensor(self.descriptor_matrix[idx], dtype=torch.float)

        # 3) Target
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "knowledge_features": descriptor_feats,
        }

        if self.target_matrix is not None:
            y = torch.tensor(self.target_matrix[idx], dtype=torch.float)
            item["labels"] = y

        return item


def compute_receptor_binding_descriptors(smiles: str):
    """
    Computes the specified descriptors for a given SMILES string.
    Returns a dictionary with descriptor names as keys.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logging.warning(f"Invalid SMILES encountered: {smiles}")
        return {
            "MolWt": np.nan,
            "MolLogP": np.nan,
            "NumHDonors": np.nan,
            "NumHAcceptors": np.nan,
        }

    try:
        descriptors = {
            "MolWt": Descriptors.MolWt(mol),
            "MolLogP": Descriptors.MolLogP(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        }
        logging.debug(f"Computed descriptors for SMILES {smiles}: {descriptors}")
        return descriptors
    except Exception as e:
        logging.error(f"Error computing descriptors for SMILES {smiles}: {e}")
        return {
            "MolWt": np.nan,
            "MolLogP": np.nan,
            "NumHDonors": np.nan,
            "NumHAcceptors": np.nan,
        }


# Custom Model Class
class KnowledgeAugmentedModel(nn.Module):
    def __init__(self, base_model, knowledge_dim, num_labels):
        super(KnowledgeAugmentedModel, self).__init__()
        self.base_model = base_model
        self.knowledge_fc = nn.Sequential(
            nn.Linear(knowledge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        hidden_size = base_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size + 64, num_labels)

    def forward(
        self, input_ids=None, attention_mask=None, knowledge_features=None, labels=None
    ):
        outputs = self.base_model.roberta(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        knowledge_output = self.knowledge_fc(knowledge_features)
        combined_output = torch.cat((pooled_output, knowledge_output), dim=1)
        logits = self.classifier(combined_output)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

    def save_pretrained(self, save_directory):
        """
        Save both the base model and custom layers' state dict.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the base model
        self.base_model.save_pretrained(save_directory)

        # Save the custom layers' state_dict
        custom_state_path = os.path.join(save_directory, "custom_model_state.pt")
        torch.save(self.state_dict(), custom_state_path)
        logging.info(f"Custom model state saved to {custom_state_path}")

    @classmethod
    def from_pretrained(cls, load_directory, knowledge_dim, num_labels):
        """
        Load both the base model and custom layers' state dict.
        """
        # Load the base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            load_directory, num_labels=num_labels, problem_type="regression"
        )

        # Initialize the custom model
        model = cls(base_model, knowledge_dim, num_labels)

        # Load the custom layers' state_dict
        custom_state_path = os.path.join(load_directory, "custom_model_state.pt")
        if os.path.exists(custom_state_path):
            model.load_state_dict(torch.load(custom_state_path))
            logging.info(f"Custom model state loaded from {custom_state_path}")
        else:
            logging.warning(
                "Custom model state file not found. Initializing custom layers with random weights."
            )

        return model


class KnowledgeAugmentedEmbeddingWrapper(nn.Module):
    """
    Wraps your KnowledgeAugmentedModel to accept 'embedded_inputs' (float)
    instead of integer input_ids, so Captum can do integrated gradients properly.
    """

    def __init__(self, knowledge_model: KnowledgeAugmentedModel):
        super().__init__()
        self.knowledge_model = knowledge_model
        # We'll grab the roberta embedding module
        self.embedding = self.knowledge_model.base_model.roberta.embeddings

    def forward(self, embedded_inputs, knowledge_features, attention_mask=None):
        """
        embedded_inputs: shape (batch_size, seq_len, hidden_dim)
        knowledge_features: shape (batch_size, knowledge_dim)
        attention_mask: shape (batch_size, seq_len)
        """
        # Pass these embedded inputs to the Roberta encoder
        # This is equivalent to roberta(...) with inputs_embeds=embedded_inputs
        encoder_outputs = self.knowledge_model.base_model.roberta(
            inputs_embeds=embedded_inputs,
            attention_mask=attention_mask,
        )
        # The last_hidden_state is shape (batch_size, seq_len, hidden_dim)
        last_hidden_state = encoder_outputs.last_hidden_state

        # Same logic as your KnowledgeAugmentedModel forward:
        pooled_output = last_hidden_state[:, 0, :]  # [CLS] token
        knowledge_output = self.knowledge_model.knowledge_fc(knowledge_features)
        combined_output = torch.cat((pooled_output, knowledge_output), dim=1)
        logits = self.knowledge_model.classifier(combined_output)
        return logits


NUM_LABELS = 3  # Adjust based on your dataset


# Load Pre-trained Model and Tokenizer
# In load_model_and_tokenizer()
def load_model_and_tokenizer():
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    model_path = "./fine_tuned_chemberta_with_knowledge"

    if os.path.exists(model_path) and os.path.isdir(model_path):
        try:
            logging.info("Attempting to load the fine-tuned model.")
            # Load config.json to get target_columns and descriptor_columns
            config_path = os.path.join(model_path, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            target_cols = config["target_columns"]
            descriptor_cols = config["descriptor_columns"]
            model_local = KnowledgeAugmentedModel.from_pretrained(
                model_path,
                knowledge_dim=len(descriptor_cols),
                num_labels=len(target_cols),
            )
            model_local.to(device)
            model_local.eval()
            model = model_local
            logging.info("Fine-tuned model loaded successfully.")
        except Exception as e:
            logging.error(
                f"Failed to load fine-tuned model: {e}. Falling back to base model."
            )
            base_model = AutoModelForSequenceClassification.from_pretrained(
                "seyonec/ChemBERTa-zinc-base-v1",
                num_labels=NUM_LABELS,
                problem_type="regression",
            )
            model_local = KnowledgeAugmentedModel(
                base_model, knowledge_dim=4, num_labels=NUM_LABELS
            )
            model_local.to(device)
            model_local.eval()
            model = model_local
            logging.info("Base pre-trained model loaded successfully.")
    else:
        logging.info(
            "Fine-tuned model directory not found. Loading base pre-trained model."
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "seyonec/ChemBERTa-zinc-base-v1",
            num_labels=NUM_LABELS,
            problem_type="regression",
        )
        model_local = KnowledgeAugmentedModel(
            base_model, knowledge_dim=4, num_labels=NUM_LABELS
        )
        model_local.to(device)
        model_local.eval()
        model = model_local
        logging.info("Base pre-trained model loaded successfully.")


# Background thread for training the model
# def train_model_in_background(file_path):s
#     global training_status, model
#     try:
#         # Update training status
#         training_status.update(
#             {
#                 "status": "running",
#                 "message": "Training started.",
#                 "progress": 0,
#                 "eta": None,
#             }
#         )
#         logging.info("Training started.")

#         # Build a descriptor CSV
#         descriptor_csv = file_path.replace(".csv", "_with_descriptors.csv")
#         create_descriptor_csv(file_path, descriptor_csv)

#         # Read the descriptor CSV
#         df = pd.read_csv(descriptor_csv)

#         # Identify target columns (exclude 'SMILES' and descriptor columns)
#         target_cols = [
#             col for col in df.columns if col not in ["SMILES"] + descriptor_names
#         ]
#         if not target_cols:
#             raise ValueError("No target columns found! Please check your CSV.")

#         logging.info(f"Detected target columns: {target_cols}")

#         # Train/Val Split
#         from sklearn.model_selection import train_test_split

#         train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

#         # Build Datasets
#         train_dataset = DescriptorDataset(
#             df=train_df,
#             tokenizer=tokenizer,
#             target_columns=target_cols,
#             descriptor_columns=descriptor_names,
#         )
#         val_dataset = DescriptorDataset(
#             df=val_df,
#             tokenizer=tokenizer,
#             target_columns=target_cols,
#             descriptor_columns=descriptor_names,
#         )

#         # Define TrainingArguments
#         training_args = TrainingArguments(
#             output_dir="./results",
#             evaluation_strategy="epoch",
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#             logging_dir="./logs",
#             logging_steps=10,
#             report_to="none",
#         )

#         # Compute total steps for progress tracking
#         total_train_size = len(train_dataset)
#         steps_per_epoch = math.ceil(
#             total_train_size / training_args.per_device_train_batch_size
#         )
#         total_steps = steps_per_epoch * training_args.num_train_epochs
#         training_status["total_steps"] = total_steps

#         # Define a custom ProgressCallback
#         class ProgressCallback(TrainerCallback):
#             def __init__(self, total_steps):
#                 self.total_steps = total_steps
#                 self.start_time = None

#             def on_train_begin(self, args, state, control, **kwargs):
#                 self.start_time = time.time()
#                 logging.info("Training callback started.")

#             def on_log(self, args, state, control, logs=None, **kwargs):
#                 if state.is_local_process_zero:
#                     current_step = state.global_step
#                     progress = (
#                         (current_step / self.total_steps) * 100
#                         if self.total_steps
#                         else 0
#                     )
#                     elapsed_time = time.time() - self.start_time
#                     steps_per_sec = (
#                         current_step / elapsed_time if elapsed_time > 0 else 0
#                     )
#                     steps_remaining = self.total_steps - current_step
#                     eta_seconds = (
#                         steps_remaining / steps_per_sec if steps_per_sec > 0 else None
#                     )
#                     eta = (
#                         time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
#                         if eta_seconds
#                         else None
#                     )

#                     training_status.update(
#                         {
#                             "progress": progress,
#                             "message": f"Training in progress... Step {current_step}/{self.total_steps}",
#                             "eta": eta,
#                         }
#                     )

#             def on_train_end(self, args, state, control, **kwargs):
#                 training_status.update(
#                     {
#                         "progress": 100,
#                         "message": "Training completed successfully.",
#                         "status": "completed",
#                         "eta": "00:00:00",
#                     }
#                 )
#                 logging.info("Training completed successfully.")

#         # Create a Trainer with the callback
#         trainer = Trainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=val_dataset,
#             callbacks=[ProgressCallback(total_steps)],
#         )

#         # Train the model
#         trainer.train()

#         # Save the fine-tuned model
#         fine_tuned_model_path = "./fine_tuned_chemberta_with_knowledge"
#         model.save_pretrained(fine_tuned_model_path)
#         logging.info(f"Fine-tuned model saved to {fine_tuned_model_path}")

#         # Save configuration
#         config = {"target_columns": target_cols, "descriptor_columns": descriptor_names}
#         with open(os.path.join(fine_tuned_model_path, "config.json"), "w") as f:
#             json.dump(config, f)
#         logging.info("Model configuration saved.")

#         # Reload the trained model
#         with model_lock:
#             knowledge_dim = len(descriptor_names)
#             num_labels = len(target_cols)
#             model = KnowledgeAugmentedModel.from_pretrained(
#                 fine_tuned_model_path,
#                 knowledge_dim=knowledge_dim,
#                 num_labels=num_labels,
#             )
#             model.to(device)
#             model.eval()

#         logging.info("Trained model reloaded into memory.")
#         training_status.update(
#             {"status": "completed", "message": "Training completed successfully."}
#         )

#     except Exception as e:
#         training_status.update(
#             {"status": "error", "message": str(e), "progress": 0, "eta": None}
#         )
#         logging.error(f"Training error: {e}")


def explain_shap_as_json(file_path):
    try:
        # Load predictions
        predictions_file_path = os.path.join(UPLOAD_FOLDER, "predictions.csv")
        predictions_df = pd.read_csv(predictions_file_path)
        knowledge_features = predictions_df[
            ["Predicted_pIC50", "Predicted_logP", "Predicted_num_atoms"]
        ].values

        # Reduce sample size for SHAP
        sampled_knowledge_features = knowledge_features[
            :100
        ]  # Use only the first 100 samples

        # Choose a representative SMILES string (e.g., the first one)
        representative_smiles = predictions_df["SMILES"].iloc[0]
        tokenized = tokenizer(
            representative_smiles,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        fixed_input_ids = tokenized["input_ids"].to(device)
        fixed_attention_mask = tokenized["attention_mask"].to(device)

        # SHAP explanation
        def model_predict(knowledge_features_input):
            knowledge_features_tensor = torch.tensor(
                knowledge_features_input, dtype=torch.float
            ).to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids=fixed_input_ids.repeat(
                        knowledge_features_tensor.shape[0], 1
                    ),
                    attention_mask=fixed_attention_mask.repeat(
                        knowledge_features_tensor.shape[0], 1
                    ),
                    knowledge_features=knowledge_features_tensor,
                )
                return outputs["logits"].cpu().numpy()

        # Use k-means to summarize the background data
        background = shap.kmeans(sampled_knowledge_features, 10)
        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(sampled_knowledge_features)

        # Format SHAP values for JSON response
        response = {
            "features": ["Predicted_pIC50", "Predicted_logP", "Predicted_num_atoms"],
            "shap_values": [sv.tolist() for sv in shap_values],
            "base_values": explainer.expected_value.tolist(),
        }
        return response

    except Exception as e:
        logging.error(f"SHAP explanation error: {str(e)}")
        return {"error": str(e)}


shap_status = {"status": "idle", "result": None, "message": ""}


def background_shap_computation(file_path):
    global shap_status
    try:
        shap_status["status"] = "running"
        shap_status["message"] = "SHAP explanation is being processed..."

        # Perform SHAP computation
        shap_result = explain_shap_as_json(file_path)

        shap_status["status"] = "completed"
        shap_status["result"] = shap_result
        shap_status["message"] = "SHAP explanation completed successfully."

    except Exception as e:
        shap_status["status"] = "error"
        shap_status["message"] = f"Error during SHAP explanation: {str(e)}"


@app.route("/start_shap_explanation", methods=["POST"])
def start_shap_explanation():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file found in request"}), 400

        file = request.files["file"]
        file_path = os.path.join(
            UPLOAD_FOLDER, f"shap_{int(time.time())}_{file.filename}"
        )
        file.save(file_path)

        # Start SHAP computation in the background
        thread = Thread(target=background_shap_computation, args=(file_path,))
        thread.start()

        return (
            jsonify({"message": "SHAP explanation started.", "status": "running"}),
            202,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/shap_status", methods=["GET"])
def get_shap_status():
    """
    Returns the current status of the SHAP computation.
    """
    global shap_status
    return jsonify(shap_status)


def explain_lime_as_json(file_path):
    try:
        # Load predictions
        predictions_file_path = os.path.join(UPLOAD_FOLDER, "predictions.csv")
        predictions_df = pd.read_csv(predictions_file_path)

        # Ensure required columns exist
        required_columns = [
            "SMILES",
            "Predicted_pIC50",
            "Predicted_logP",
            "Predicted_num_atoms",
        ]
        for col in required_columns:
            if col not in predictions_df.columns:
                raise ValueError(f"Column '{col}' is missing from predictions.csv")

        knowledge_features = predictions_df[
            ["Predicted_pIC50", "Predicted_logP", "Predicted_num_atoms"]
        ].values

        # Choose a representative instance (e.g., the first one)
        representative_smiles = predictions_df["SMILES"].iloc[0]
        tokenized = tokenizer(
            representative_smiles,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        fixed_input_ids = tokenized["input_ids"].to(device)
        fixed_attention_mask = tokenized["attention_mask"].to(device)
        representative_features = knowledge_features[0].reshape(1, -1)

        # Define feature names and types
        feature_names = ["Predicted_pIC50", "Predicted_logP", "Predicted_num_atoms"]

        # Initialize LIME explainer
        explainer = LimeTabularExplainer(
            training_data=knowledge_features,
            feature_names=feature_names,
            mode="regression",
            discretize_continuous=True,
        )

        # Define the prediction function for LIME
        def model_predict(knowledge_features_input):
            knowledge_features_tensor = (
                torch.tensor(knowledge_features_input).float().to(device)
            )
            with torch.no_grad():
                # Repeat fixed_input_ids and fixed_attention_mask for the batch size
                batch_size = knowledge_features_tensor.shape[0]
                repeated_input_ids = fixed_input_ids.repeat(batch_size, 1)
                repeated_attention_mask = fixed_attention_mask.repeat(batch_size, 1)

                outputs = model(
                    input_ids=repeated_input_ids,
                    attention_mask=repeated_attention_mask,
                    knowledge_features=knowledge_features_tensor,
                )
                return outputs["logits"].cpu().numpy()

        # Generate LIME explanation for the first instance
        lime_exp = explainer.explain_instance(
            representative_features[0], model_predict, num_features=3, num_samples=500
        )

        # Extract explanation as a dictionary
        explanation = {"feature_names": feature_names, "weights": lime_exp.as_list()}

        return explanation

    except Exception as e:
        logging.error(f"LIME explanation error: {str(e)}")
        return {"error": str(e)}


lime_status = {"status": "idle", "result": None, "message": ""}


def background_lime_computation(file_path):
    global lime_status
    try:
        lime_status["status"] = "running"
        lime_status["message"] = "LIME explanation is being processed..."

        # Perform LIME computation
        lime_result = explain_lime_as_json(file_path)

        lime_status["status"] = "completed"
        lime_status["result"] = lime_result
        lime_status["message"] = "LIME explanation completed successfully."

    except Exception as e:
        lime_status["status"] = "error"
        lime_status["message"] = f"Error during LIME explanation: {str(e)}"
        logging.error(f"LIME explanation error: {str(e)}")


@app.route("/start_lime_explanation", methods=["POST"])
def start_lime_explanation():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file found in request"}), 400

        file = request.files["file"]
        file_path = os.path.join(
            UPLOAD_FOLDER, f"lime_{int(time.time())}_{file.filename}"
        )
        file.save(file_path)

        # Start LIME computation in the background
        thread = Thread(target=background_lime_computation, args=(file_path,))
        thread.start()

        return (
            jsonify({"message": "LIME explanation started.", "status": "running"}),
            202,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/lime_status", methods=["GET"])
def get_lime_status():
    """
    Returns the current status of the LIME computation.
    """
    global lime_status
    return jsonify(lime_status)


@app.route("/lime_explanation", methods=["GET"])
def get_lime_explanation():
    """
    Returns the LIME explanation result.
    """
    global lime_status
    if lime_status["status"] == "completed":
        return jsonify(lime_status["result"]), 200
    elif lime_status["status"] == "error":
        return jsonify({"error": lime_status["message"]}), 500
    else:
        return jsonify({"message": "LIME explanation is not yet completed."}), 202


# Route to start training
@app.route("/train", methods=["POST"])
def train_model():
    try:
        # Save the uploaded file
        if "file" not in request.files:
            return jsonify({"error": "No file found in request"}), 400
        file = request.files["file"]
        file_path = os.path.join(
            UPLOAD_FOLDER, f"train_{int(time.time())}_{file.filename}"
        )
        file.save(file_path)

        # Start background training
        thread = Thread(target=train_model_in_background, args=(file_path,))
        thread.start()

        # Respond immediately
        return (
            jsonify(
                {
                    "message": "Training started in the background. Check status at /status."
                }
            ),
            202,
        )

    except Exception as e:
        logging.error(f"Error during training setup: {e}")
        return jsonify({"error": str(e)}), 500


# Route to check training status
@app.route("/get_training_status", methods=["GET"])
def get_training_status():
    return jsonify(training_status)


def run_predictions(file_path):
    global prediction_status
    try:
        # Set status to 'running'
        prediction_status.update(
            {
                "status": "running",
                "message": "Prediction started.",
                "progress": 0,
                "eta": None,
            }
        )
        logging.info("Prediction started.")

        # Load prediction data
        smiles_predict, _, knowledge_features_predict = load_smiles_data(
            file_path, properties_present=False
        )

        # Prepare the dataset
        predict_dataset = SMILESDataset(
            smiles_list=smiles_predict,
            knowledge_features=knowledge_features_predict,
            tokenizer=tokenizer,
            max_length=128,
        )

        # Create DataLoader
        dataloader = DataLoader(predict_dataset, batch_size=16)

        # Initialize variables for progress tracking
        total_steps = len(dataloader)
        prediction_status["total_steps"] = total_steps
        start_time = time.time()

        predictions = []
        model.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataloader, desc="Predicting")):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs["logits"]
                predictions.extend(logits.cpu().numpy())

                # Update progress
                current_step = step + 1
                progress = (current_step / total_steps) * 100
                elapsed_time = time.time() - start_time
                steps_per_sec = current_step / elapsed_time if elapsed_time > 0 else 0
                steps_remaining = total_steps - current_step
                eta_seconds = (
                    steps_remaining / steps_per_sec if steps_per_sec > 0 else None
                )
                eta = (
                    time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                    if eta_seconds
                    else None
                )

                prediction_status.update(
                    {
                        "progress": progress,
                        "message": f"Prediction in progress... {current_step}/{total_steps}",
                        "eta": eta,
                    }
                )

        # Save predictions to CSV
        predictions_df = pd.DataFrame(
            predictions,
            columns=["Predicted_pIC50", "Predicted_logP", "Predicted_num_atoms"],
        )
        predictions_df["SMILES"] = smiles_predict

        # Add descriptors to the DataFrame
        predictions_df["MolWt"] = [kf[0] for kf in knowledge_features_predict]
        predictions_df["MolLogP"] = [kf[1] for kf in knowledge_features_predict]
        predictions_df["NumHDonors"] = [kf[2] for kf in knowledge_features_predict]
        predictions_df["NumHAcceptors"] = [kf[3] for kf in knowledge_features_predict]

        # Reorder columns to match the desired order
        predictions_df = predictions_df[
            [
                "Predicted_pIC50",
                "Predicted_logP",
                "Predicted_num_atoms",
                "MolWt",
                "MolLogP",
                "NumHDonors",
                "NumHAcceptors",
                "SMILES",
            ]
        ]

        # Save to CSV
        predictions_file = os.path.join(UPLOAD_FOLDER, "predictions.csv")
        predictions_df.to_csv(predictions_file, index=False)

        # Update status to 'completed'
        prediction_status.update(
            {
                "status": "completed",
                "message": f"Prediction completed successfully. Results saved to {predictions_file}.",
                "progress": 100,
                "eta": "00:00:00",
            }
        )
        logging.info("Prediction completed.")

        # Automatically start SHAP and LIME explanations
        logging.info("Starting SHAP and LIME explanations.")
        Thread(target=background_shap_computation, args=(predictions_file,)).start()
        Thread(target=background_lime_computation, args=(predictions_file,)).start()
        logging.info("SHAP and LIME explanations have been initiated.")

    except Exception as e:
        # Update status to 'error'
        prediction_status.update(
            {"status": "error", "message": str(e), "progress": 0, "eta": None}
        )
        logging.error(f"Prediction error: {e}")


# Route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    if training_status["status"] == "running":
        return jsonify({"error": "Model is being trained. Try again later."}), 400

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file found in request"}), 400

        file = request.files["file"]
        file_path = os.path.join(
            UPLOAD_FOLDER, f"predict_{int(time.time())}_{file.filename}"
        )
        file.save(file_path)

        # Start the prediction in a background thread
        thread = Thread(target=run_predictions, args=(file_path,))
        thread.start()

        # Return an immediate response
        return (
            jsonify(
                {
                    "message": "Prediction started in the background. Check status at /get_prediction_status."
                }
            ),
            202,
        )

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    try:
        # Get pagination parameters from the query string
        page = int(request.args.get("page", 1))  # Default to page 1 if not provided
        limit = int(request.args.get("limit", 5))  # Default to 5 records per page

        predictions_file_path = os.path.join(UPLOAD_FOLDER, "predictions.csv")

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(predictions_file_path)

        # Optionally, rename or rearrange columns if needed
        df = df[["Predicted_pIC50", "Predicted_logP", "Predicted_num_atoms", "SMILES"]]

        # Calculate the start and end indices for slicing the data
        start = (page - 1) * limit
        end = start + limit

        # Slice the DataFrame based on the pagination parameters
        paginated_data = df.iloc[start:end]

        # Convert the DataFrame to a list of dictionaries (JSON-compatible format)
        predictions = paginated_data.to_dict(orient="records")

        # Calculate total records and total pages
        total_records = len(df)
        total_pages = (total_records // limit) + (
            1 if total_records % limit != 0 else 0
        )

        # Return predictions with pagination metadata
        return jsonify(
            {
                "predictions": predictions,
                "pagination": {
                    "total_records": total_records,
                    "total_pages": total_pages,
                    "current_page": page,
                    "limit": limit,
                },
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def explain_shap_in_background(file_path):
    global shap_status
    try:
        shap_status["status"] = "running"
        shap_status["message"] = "SHAP explanation is being processed..."

        # Perform SHAP computation
        shap_result = explain_shap_as_json(file_path)

        if "error" in shap_result:
            shap_status["status"] = "error"
            shap_status["message"] = shap_result["error"]
            return

        # Generate and save the SHAP summary plot
        shap_values = np.array(shap_result["shap_values"])
        base_values = np.array(shap_result["base_values"])
        features = shap_result["features"]

        # Convert shap_values to a numpy array
        shap_values = np.array(shap_values)

        # Create a directory to save SHAP plots if it doesn't exist
        shap_plots_dir = os.path.join("shap_plots")
        os.makedirs(shap_plots_dir, exist_ok=True)

        # Generate a unique filename for the plot
        plot_filename = f"shap_summary_{int(time.time())}.png"
        plot_path = os.path.join(shap_plots_dir, plot_filename)

        # Create the summary plot and save it
        plt.figure()
        shap.summary_plot(shap_values, features=features, show=False)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        # Update the SHAP status with the plot path
        shap_status["status"] = "completed"
        shap_status["result"] = {
            "shap_values": shap_result["shap_values"],
            "base_values": shap_result["base_values"],
            "plot_filename": plot_filename,
        }
        shap_status["message"] = (
            "SHAP explanation and summary plot completed successfully."
        )

        logging.info(f"SHAP summary plot saved to {plot_path}")

    except Exception as e:
        shap_status["status"] = "error"
        shap_status["message"] = f"Error during SHAP explanation: {str(e)}"
        logging.error(f"SHAP explanation error: {str(e)}")


@app.route("/download_shap_plot/<plot_filename>", methods=["GET"])
def download_shap_plot(plot_filename):
    """
    Endpoint to download the SHAP summary plot image.
    """
    try:
        shap_plots_dir = os.path.join("shap_plots")
        file_path = os.path.join(shap_plots_dir, plot_filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "Plot file not found"}), 404

        return send_from_directory(
            directory=shap_plots_dir,
            path=plot_filename,
            as_attachment=True,
            mimetype="image/png",
        )
    except Exception as e:
        logging.error(f"Error in /download_shap_plot/{plot_filename}: {e}")
        return jsonify({"error": str(e)}), 500


# Route for SHAP explanation (this starts the process in the background)
@app.route("/explain_predictions", methods=["POST"])
def explain_predictions():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file found in request"}), 400

        # Save the uploaded file to the server
        file = request.files["file"]
        file_path = os.path.join(
            UPLOAD_FOLDER, f"predict_{int(time.time())}_{file.filename}"
        )
        file.save(file_path)

        # Start the SHAP explanation in a background thread
        thread = Thread(target=explain_shap_in_background, args=(file_path,))
        thread.start()

        # Immediately respond with a 200 OK indicating file was received
        return (
            jsonify(
                {"message": "File received and processing started in the background."}
            ),
            200,
        )

    except Exception as e:
        logging.error(f"Error during SHAP explanation setup: {e}")
        return jsonify({"error": str(e)}), 500


# Route to check the status of SHAP explanation
@app.route("/get_explanation_status", methods=["GET"])
def get_explanation_status():
    return jsonify(training_status)


import requests  # Add this import at the top of your script if you plan to use HTTP requests


def train_model_in_background(file_path):
    global training_status, model
    try:
        # Set initial status
        training_status.update(
            {
                "status": "running",
                "message": "Training started.",
                "progress": 0,
                "eta": None,
            }
        )
        logging.info("Training started.")

        # Load training data
        smiles_train, targets_train, knowledge_features_train = load_smiles_data(
            file_path, properties_present=True
        )

        if targets_train is None or len(targets_train) == 0:
            raise ValueError("No target properties found after processing the dataset.")

        # Split into training and validation sets
        from sklearn.model_selection import train_test_split

        (
            smiles_train,
            smiles_val,
            targets_train,
            targets_val,
            features_train,
            features_val,
        ) = train_test_split(
            smiles_train,
            targets_train,
            knowledge_features_train,
            test_size=0.2,
            random_state=42,
        )

        # Prepare datasets
        train_dataset = SMILESDataset(
            smiles_list=smiles_train,
            knowledge_features=features_train,
            target_list=targets_train,
            tokenizer=tokenizer,
            max_length=128,
        )
        eval_dataset = SMILESDataset(
            smiles_list=smiles_val,
            knowledge_features=features_val,
            target_list=targets_val,
            tokenizer=tokenizer,
            max_length=128,
        )

        # Define TrainingArguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            report_to="none",
            fp16=True,  # Optional: Enable mixed precision for efficiency
        )

        # Compute total steps for progress tracking
        total_train_size = len(train_dataset)
        steps_per_epoch = math.ceil(
            total_train_size / training_args.per_device_train_batch_size
        )
        total_steps = steps_per_epoch * training_args.num_train_epochs
        training_status["total_steps"] = total_steps

        # Define a custom ProgressCallback
        class ProgressCallback(TrainerCallback):
            def __init__(self, total_steps):
                self.total_steps = total_steps
                self.start_time = None

            def on_train_begin(self, args, state, control, **kwargs):
                self.start_time = time.time()
                logging.info("Training callback started.")

            def on_log(self, args, state, control, logs=None, **kwargs):
                if state.is_local_process_zero:
                    current_step = state.global_step
                    progress = (
                        (current_step / self.total_steps) * 100
                        if self.total_steps
                        else 0
                    )
                    elapsed_time = time.time() - self.start_time
                    steps_per_sec = (
                        current_step / elapsed_time if elapsed_time > 0 else 0
                    )
                    steps_remaining = self.total_steps - current_step
                    eta_seconds = (
                        steps_remaining / steps_per_sec if steps_per_sec > 0 else None
                    )
                    eta = (
                        time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                        if eta_seconds
                        else None
                    )

                    training_status.update(
                        {
                            "progress": progress,
                            "message": f"Training in progress... Step {current_step}/{self.total_steps}",
                            "eta": eta,
                        }
                    )

            def on_train_end(self, args, state, control, **kwargs):
                training_status.update(
                    {
                        "progress": 100,
                        "message": "Training completed successfully.",
                        "status": "completed",
                        "eta": "00:00:00",
                    }
                )
                logging.info("Training completed successfully.")

        # Create a Trainer with the callback
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[ProgressCallback(total_steps)],
        )

        # Train the model
        trainer.train()

        # Save the fine-tuned model
        fine_tuned_model_path = "./fine_tuned_chemberta_with_knowledge"
        model.save_pretrained(fine_tuned_model_path)
        logging.info(f"Fine-tuned model saved to {fine_tuned_model_path}")

        # Save configuration
        config = {
            "target_columns": ["pIC50", "num_atoms", "logP"],
            "descriptor_columns": ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors"],
        }
        with open(os.path.join(fine_tuned_model_path, "config.json"), "w") as f:
            json.dump(config, f)
        logging.info("Model configuration saved.")

        # Reload the trained model
        with model_lock:
            knowledge_dim = 4  # Updated from 3 to 4
            num_labels = targets_train.shape[1]
            model = KnowledgeAugmentedModel.from_pretrained(
                fine_tuned_model_path,
                knowledge_dim=knowledge_dim,
                num_labels=num_labels,
            )
            model.to(device)
            model.eval()

        logging.info("Trained model reloaded into memory.")

    except Exception as e:
        # Handle exceptions
        training_status.update(
            {"status": "error", "message": str(e), "progress": 0, "eta": None}
        )
        logging.error(f"Training error: {e}")


@app.route("/get_prediction_status", methods=["GET"])
def get_prediction_status():
    return jsonify(prediction_status)


# Allowed file extensions
def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    """
    ALLOWED_EXTENSIONS = {"csv"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Function to calculate molecular descriptors from SMILES
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "Valid_SMILES": False,
            "MolWt": None,
            "LogP": None,
            "NumRotatableBonds": None,
            "NumAtoms": None,
            "NumHDonors": None,
            "NumHAcceptors": None,
        }
    return {
        "Valid_SMILES": True,
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAtoms": mol.GetNumAtoms(),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
    }


# Global dictionary to store task statuses and results
tasks = {}


def process_calculation(task_id, file_path):
    """
    Background thread function to process the CSV file and calculate molecular properties.
    """
    try:
        logging.info(f"Task {task_id}: Starting calculation.")
        tasks[task_id]["status"] = "running"

        # Read the uploaded CSV file
        df = pd.read_csv(file_path)

        # Check if 'SMILES' column exists
        if "SMILES" not in df.columns:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["message"] = "CSV must contain a 'SMILES' column."
            logging.error(f"Task {task_id}: 'SMILES' column missing.")
            return

        # Calculate descriptors
        descriptors = df["SMILES"].apply(calculate_descriptors)

        # Convert descriptors to a DataFrame
        descriptors_df = pd.DataFrame(descriptors.tolist())

        # Combine the original DataFrame with the descriptors
        combined_df = pd.concat([df, descriptors_df], axis=1)

        # Handle rows where descriptors couldn't be calculated
        invalid_rows = combined_df[combined_df["Valid_SMILES"] == False]
        if not invalid_rows.empty:
            logging.warning(
                f"Task {task_id}: Some SMILES strings could not be parsed and have NaN values."
            )

        # Save the combined DataFrame to a new CSV file
        output_filename = f"calculated_properties_{task_id}.csv"
        output_path = os.path.join("results", output_filename)
        os.makedirs("results", exist_ok=True)
        combined_df.to_csv(output_path, index=False)

        # Update task status and result
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = output_filename
        tasks[task_id]["message"] = "Calculation completed successfully."
        logging.info(f"Task {task_id}: Calculation completed.")

    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["message"] = str(e)
        logging.error(f"Task {task_id}: Error during calculation - {e}")


@app.route("/calculate_properties_async", methods=["POST"])
def calculate_properties_async():
    """
    Asynchronous endpoint to calculate molecular properties from a CSV of SMILES strings.
    Accepts a CSV file with a 'SMILES' column.
    Returns a unique task_id to track the calculation.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected for uploading"}), 400

        if file and allowed_file(file.filename):
            # Generate a unique task ID
            task_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()

            # Save the uploaded file
            upload_filename = f"{task_id}_{file.filename}"
            upload_path = os.path.join("uploads", upload_filename)
            os.makedirs("uploads", exist_ok=True)
            file.save(upload_path)

            # Initialize task in the tasks dictionary
            tasks[task_id] = {
                "status": "pending",
                "message": "Task is pending and will start shortly.",
                "result": None,
                "uploaded_at": timestamp,
            }

            # Start the background thread
            thread = Thread(target=process_calculation, args=(task_id, upload_path))
            thread.start()

            logging.info(f"Task {task_id}: Received and started.")

            return jsonify({"task_id": task_id}), 202  # 202 Accepted

        else:
            return jsonify({"error": "Allowed file types are csv"}), 400

    except Exception as e:
        logging.error(f"Error in /calculate_properties_async: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/calculate_properties_status/<task_id>", methods=["GET"])
def calculate_properties_status(task_id):
    """
    Endpoint to check the status of a molecular properties calculation task.
    If completed, provides a link to download the results.
    """
    try:
        if task_id not in tasks:
            return jsonify({"error": "Invalid task ID"}), 404

        task = tasks[task_id]

        response = {
            "task_id": task_id,
            "status": task["status"],
            "message": task["message"],
        }

        if task["status"] == "completed":
            # Provide a download link or the filename
            response["download_url"] = f"/download_results/{task['result']}"
        elif task["status"] == "error":
            response["error_detail"] = task["message"]

        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Error in /calculate_properties_status/{task_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/download_results/<filename>", methods=["GET"])
def download_results(filename):
    """
    Endpoint to download the calculated properties CSV file.
    """
    try:
        file_path = os.path.join("results", filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        return send_file(
            file_path, mimetype="text/csv", as_attachment=True, download_name=filename
        )

    except Exception as e:
        logging.error(f"Error in /download_results/{filename}: {e}")
        return jsonify({"error": str(e)}), 500


def compute_integrated_gradients_embedding(
    knowledge_model: KnowledgeAugmentedModel,
    input_ids: torch.Tensor,
    knowledge_features: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    target_idx=0,
):
    try:
        # Wrap the model
        wrapper = KnowledgeAugmentedEmbeddingWrapper(knowledge_model).to(device)

        # Convert input_ids to embeddings
        embedded_inputs = wrapper.embedding.word_embeddings(input_ids)
        logging.info(f"Embedded inputs shape: {embedded_inputs.shape}")

        # Initialize Integrated Gradients
        from captum.attr import IntegratedGradients

        def wrapper_forward(embeds):
            logits = wrapper(
                embedded_inputs=embeds,
                knowledge_features=knowledge_features,
                attention_mask=attention_mask,
            )
            logging.info(f"Logits shape in wrapper_forward: {logits.shape}")
            # Ensure target_idx is within bounds
            if target_idx >= logits.shape[1]:
                raise ValueError(
                    f"target_idx {target_idx} is out of bounds for logits with shape {logits.shape}"
                )
            return logits[:, target_idx]

        ig = IntegratedGradients(wrapper_forward)

        # Define baselines
        baseline_embeds = torch.zeros_like(embedded_inputs).to(device)
        logging.info(f"Baseline embeddings shape: {baseline_embeds.shape}")

        # Compute attributions
        attributions, delta = ig.attribute(
            inputs=embedded_inputs,
            baselines=baseline_embeds,
            n_steps=50,
            return_convergence_delta=True,
        )

        logging.info(f"Attributions computed shape: {attributions.shape}")
        logging.info(f"Convergence delta: {delta}")

        return attributions, delta

    except Exception as e:
        logging.error(
            f"Error in compute_integrated_gradients_embedding: {e}", exc_info=True
        )
        raise e


@app.route("/api/actual_vs_predicted_dynamic", methods=["GET"])
def actual_vs_predicted_dynamic():
    """
    A dynamic endpoint that calculates actual vs predicted for each 'Predicted_<property>'
    column in 'predictions.csv', IF we have an RDKit function to compute the actual
    from SMILES. This avoids hard-coding the property names like logP/num_atoms.

    Returns (JSON):
    {
      "properties": {
        "logP": {
          "data": [
            {
              "SMILES": "CCOC(=O)c1ccccc1",
              "actual": 2.128,
              "predicted": 1.987
            },
            ...
          ],
          "mse": ...,
          "mae": ...,
          "r2": ...
        },
        "num_atoms": {
          "data": [...],
          "mse": ...,
          "mae": ...,
          "r2": ...
        },
        ...
      }
    }
    """
    try:
        predictions_file_path = os.path.join(UPLOAD_FOLDER, "predictions.csv")
        if not os.path.exists(predictions_file_path):
            return (
                jsonify({"error": "predictions.csv not found in uploads folder."}),
                404,
            )

        df = pd.read_csv(predictions_file_path)

        # 1) Define a dictionary that maps property suffix -> RDKit function
        #    to compute the 'actual' from the SMILES.
        #    - You can add more if needed.
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        property_functions = {
            "logP": lambda mol: Descriptors.MolLogP(mol),
            "num_atoms": lambda mol: mol.GetNumAtoms(),
            # e.g., "molwt": lambda mol: Descriptors.MolWt(mol),
            # If you want more properties, you can add them here
        }

        # 2) Identify all columns in df that are predicted properties:
        #    i.e., columns that start with "Predicted_"
        predicted_columns = [col for col in df.columns if col.startswith("Predicted_")]

        # We'll store results in a dictionary like:
        # {
        #   "properties": {
        #       "logP": {
        #           "data": [...],
        #           "mse": ...,
        #           "mae": ...,
        #           "r2": ...
        #       },
        #       ...
        #   }
        # }
        results = {"properties": {}}

        # We'll need arrays to compute metrics per property
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # 3) For each "Predicted_<suffix>" column, see if <suffix> is in property_functions
        for pred_col in predicted_columns:
            # e.g., pred_col = "Predicted_logP" => suffix = "logP"
            suffix = pred_col.replace("Predicted_", "").strip()
            if suffix not in property_functions:
                # We have no RDKit function for this property, skip
                continue

            # We'll collect data points { SMILES, actual, predicted } in a list
            data_points = []
            actual_vals = []
            predicted_vals = []

            # We'll define the RDKit function from our dictionary
            rdkit_func = property_functions[suffix]

            # Iterate over the DataFrame
            for idx, row in df.iterrows():
                if "SMILES" not in row:
                    # If there's no SMILES column, we cannot compute actual. Skip.
                    continue

                smiles = row["SMILES"]
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    # Invalid SMILES, skip or handle as needed
                    continue

                # Compute actual value via RDKit
                actual_val = rdkit_func(mol)

                # Get predicted value
                predicted_val = row[pred_col]

                # Collect them
                data_points.append(
                    {
                        "SMILES": smiles,
                        "actual": float(actual_val),
                        "predicted": float(predicted_val),
                    }
                )

                actual_vals.append(actual_val)
                predicted_vals.append(predicted_val)

            # Once we gather all rows for this property, compute metrics
            mse_ = (
                mean_squared_error(actual_vals, predicted_vals)
                if len(actual_vals) > 1
                else None
            )
            mae_ = (
                mean_absolute_error(actual_vals, predicted_vals)
                if len(actual_vals) > 1
                else None
            )
            r2_ = (
                r2_score(actual_vals, predicted_vals) if len(actual_vals) > 1 else None
            )

            # Store in results
            results["properties"][suffix] = {
                "data": data_points,
                "mse": mse_,
                "mae": mae_,
                "r2": r2_,
            }

        return jsonify(results), 200

    except Exception as e:
        logging.error(f"Error in /api/actual_vs_predicted_dynamic: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/check-training-status", methods=["GET"])
def check_training_status():
    """
    Endpoint to check if a trained model exists.
    Returns:
        JSON: { "isTrained": true } if model exists, else { "isTrained": false }
    """
    model_path = "./fine_tuned_chemberta_with_knowledge"
    required_files = ["config.json", "custom_model_state.pt"]

    if os.path.exists(model_path) and os.path.isdir(model_path):
        # Check for all required files
        files_present = all(
            os.path.exists(os.path.join(model_path, f)) for f in required_files
        )
        if files_present:
            return jsonify({"isTrained": True}), 200
    return jsonify({"isTrained": False}), 200


@app.route("/api/chart-data", methods=["GET"])
def get_chart_data():
    """
    Endpoint to retrieve chart data from predictions.csv.
    Returns:
        JSON: List of data points with 'name', 'propertyA', 'propertyB', 'propertyC'.
    """
    try:
        predictions_file_path = os.path.join(UPLOAD_FOLDER, "predictions.csv")

        if not os.path.exists(predictions_file_path):
            return jsonify({"error": "Predictions file not found."}), 404

        df = pd.read_csv(predictions_file_path)

        # Ensure the necessary columns exist
        required_columns = [
            "SMILES",
            "Predicted_pIC50",
            "Predicted_logP",
            "Predicted_num_atoms",
        ]
        for col in required_columns:
            if col not in df.columns:
                return (
                    jsonify({"error": f"Column '{col}' not found in predictions.csv."}),
                    400,
                )

        # Rename columns to match Recharts expectations
        chart_data = df.rename(
            columns={
                "SMILES": "name",
                "Predicted_pIC50": "propertyA",
                "Predicted_logP": "propertyB",
                "Predicted_num_atoms": "propertyC",
            }
        )[["name", "propertyA", "propertyB", "propertyC"]].to_dict(orient="records")

        return jsonify(chart_data), 200

    except Exception as e:
        logging.error(f"Error in /api/chart-data: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/compound-properties", methods=["GET"])  # Renamed endpoint
def get_compound_properties():
    """
    Endpoint to retrieve compound properties from predictions.csv.
    Returns:
        JSON: List of data points with 'name', 'propertyA', 'propertyB', 'propertyC'.
    """
    try:
        predictions_file_path = os.path.join(UPLOAD_FOLDER, "predictions.csv")

        if not os.path.exists(predictions_file_path):
            return jsonify({"error": "Predictions file not found."}), 404

        df = pd.read_csv(predictions_file_path)

        # Ensure the necessary columns exist
        required_columns = [
            "SMILES",
            "Predicted_pIC50",
            "Predicted_logP",
            "Predicted_num_atoms",
        ]
        for col in required_columns:
            if col not in df.columns:
                return (
                    jsonify({"error": f"Column '{col}' not found in predictions.csv."}),
                    400,
                )

        # Rename columns to match Recharts expectations
        chart_data = df.rename(
            columns={
                "SMILES": "name",
                "Predicted_pIC50": "propertyA",
                "Predicted_logP": "propertyB",
                "Predicted_num_atoms": "propertyC",
            }
        )[["name", "propertyA", "propertyB", "propertyC"]].to_dict(orient="records")

        return jsonify(chart_data), 200

    except Exception as e:
        logging.error(f"Error in /api/compound-properties: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-metrics", methods=["GET"])
def get_model_metrics():
    """
    1) Reads 'predictions.csv' with columns [Predicted_pIC50, Predicted_logP, Predicted_num_atoms, SMILES].
    2) Uses RDKit to compute actual logP and actual number of atoms from the SMILES.
    3) For logP and num_atoms, computes MSE, MAE, R² (regression).
    4) For logP and num_atoms, also does a classification threshold to compute accuracy.
       - logP threshold 0 => class=1 if actual>0 else 0 (sim. for predicted).
       - num_atoms threshold 20 => class=1 if actual>=20 else 0 (sim. for predicted).
    5) For pIC50, we have no actual data from SMILES, so by default we skip its regression metrics.
       If you want a dummy classification for pIC50>0 => class=1 else 0, we do that as well.
    6) Returns a JSON with all computed metrics per property.
    """
    try:
        # Path to predictions CSV
        csv_path = "./uploads/predictions.csv"
        if not os.path.exists(csv_path):
            return jsonify({"error": "File not found"}), 404

        df = pd.read_csv(csv_path)

        # Ensure required columns
        required_cols = [
            "Predicted_pIC50",
            "Predicted_logP",
            "Predicted_num_atoms",
            "SMILES",
        ]
        for c in required_cols:
            if c not in df.columns:
                return jsonify({"error": f"Missing column '{c}' in CSV"}), 400

        # We'll build a metrics dictionary like:
        # {
        #   "pIC50": {"mse": ..., "mae": ..., "r2": ..., "accuracy": ...},
        #   "logP": {"mse": ..., "mae": ..., "r2": ..., "accuracy": ...},
        #   "num_atoms": {"mse": ..., "mae": ..., "r2": ..., "accuracy": ...}
        # }
        results = {
            "pIC50": {"mse": None, "mae": None, "r2": None, "accuracy": None},
            "logP": {"mse": None, "mae": None, "r2": None, "accuracy": None},
            "num_atoms": {"mse": None, "mae": None, "r2": None, "accuracy": None},
        }

        # Prepare lists for logP and num_atoms regression
        actual_logp_vals = []
        predicted_logp_vals = []

        actual_num_atoms_vals = []
        predicted_num_atoms_vals = []

        # Classification label arrays
        # We'll define them for each property so we can compute accuracy
        actual_logp_labels = []
        predicted_logp_labels = []

        actual_num_atoms_labels = []
        predicted_num_atoms_labels = []

        # For pIC50, we only have predicted values, so no real regression.
        # But we can do a dummy classification if user wants.
        predicted_pic50_vals = []
        predicted_pic50_labels = []

        from rdkit import Chem
        from rdkit.Chem import Descriptors

        for idx, row in df.iterrows():
            smiles = row["SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue  # skip invalid SMILES

            # ---------- Actual Values from RDKit ----------
            actual_logp = Descriptors.MolLogP(mol)
            actual_num_atoms = mol.GetNumAtoms()

            # ---------- Predicted Values ----------
            pred_logp = row["Predicted_logP"]
            pred_num_atoms = row["Predicted_num_atoms"]
            pred_pic50 = row["Predicted_pIC50"]

            # Save them for regression
            actual_logp_vals.append(actual_logp)
            predicted_logp_vals.append(pred_logp)

            actual_num_atoms_vals.append(actual_num_atoms)
            predicted_num_atoms_vals.append(pred_num_atoms)

            # ---------- Classification: logP threshold = 0 ----------
            # Classify actual vs. predicted
            # (1 if > 0, else 0)
            actual_logp_labels.append(1 if actual_logp > 0 else 0)
            predicted_logp_labels.append(1 if pred_logp > 0 else 0)

            # ---------- Classification: num_atoms threshold = 20 ----------
            actual_num_atoms_labels.append(1 if actual_num_atoms >= 20 else 0)
            predicted_num_atoms_labels.append(1 if pred_num_atoms >= 20 else 0)

            # ---------- pIC50: only predicted available ----------
            predicted_pic50_vals.append(pred_pic50)
            # If you want a classification threshold, e.g. pIC50>0 => active:
            predicted_pic50_labels.append(1 if pred_pic50 > 0 else 0)

        # --------------- Regression Metrics for logP ---------------
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            r2_score,
            accuracy_score,
        )

        if len(actual_logp_vals) > 0:
            mse_logp = mean_squared_error(actual_logp_vals, predicted_logp_vals)
            mae_logp = mean_absolute_error(actual_logp_vals, predicted_logp_vals)
            r2_logp = r2_score(actual_logp_vals, predicted_logp_vals)
            results["logP"]["mse"] = mse_logp
            results["logP"]["mae"] = mae_logp
            results["logP"]["r2"] = r2_logp

            # Classification accuracy for logP
            acc_logp = accuracy_score(actual_logp_labels, predicted_logp_labels)
            results["logP"]["accuracy"] = acc_logp

        # --------------- Regression Metrics for num_atoms ---------------
        if len(actual_num_atoms_vals) > 0:
            mse_atoms = mean_squared_error(
                actual_num_atoms_vals, predicted_num_atoms_vals
            )
            mae_atoms = mean_absolute_error(
                actual_num_atoms_vals, predicted_num_atoms_vals
            )
            r2_atoms = r2_score(actual_num_atoms_vals, predicted_num_atoms_vals)
            results["num_atoms"]["mse"] = mse_atoms
            results["num_atoms"]["mae"] = mae_atoms
            results["num_atoms"]["r2"] = r2_atoms

            # Classification accuracy for num_atoms
            acc_atoms = accuracy_score(
                actual_num_atoms_labels, predicted_num_atoms_labels
            )
            results["num_atoms"]["accuracy"] = acc_atoms

        # --------------- pIC50 (NO real regression, but optional classification) ---------------
        # We have no actual pIC50 from SMILES, so cannot do real MSE/MAE/R².
        # If you had real pIC50 in your CSV, you'd read it and do a real regression comparison here.
        # For demonstration, let's do classification vs. 0 threshold
        # But note that "actual" is missing, so this is nonsense as a real metric. It will always
        # compare predicted to a single threshold or to itself.
        if len(predicted_pic50_vals) > 0:
            # We'll skip MSE/MAE/R² since no actual values exist
            # But do a "self-comparison" classification to show how it'd look
            # In reality, this yields 100% if you compare the same predictions to the same threshold
            # so it's not very meaningful.
            # We can do "accuracy" = how many predicted pIC50>0 vs. predicted pIC50>0 (the same!)
            # If you actually had an "Actual_pIC50" column, you'd compute real metrics.

            # We'll at least set the accuracy to the fraction that is >0 if we wanted to compare
            # But it's a dummy example:
            # predicted_pic50_labels is set above
            # actual 'labels'? We have none, so skip or do a dummy approach:
            # For demonstration, let's compare predicted vs. predicted (which = 100% accuracy).
            # We'll skip that to avoid confusion.

            results["pIC50"][
                "accuracy"
            ] = None  # or 1.0 if you literally compare the same array

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


import torch
from captum.attr import IntegratedGradients


def compute_integrated_gradients(model, input_ids, attention_mask, target_idx=0):
    """
    model: A PyTorch model that returns 'logits'
    input_ids: Tensor of shape [batch_size, seq_len]
    attention_mask: Tensor of shape [batch_size, seq_len]
    target_idx: which output dimension to compute IG for (0 if single-output regression, or an index for multi-class)
    """

    # We define a forward function for Captum
    def forward_func(input_ids_embedded):
        """
        input_ids_embedded is a float Tensor that we'll feed into the model in place of 'input_ids'.
        For typical huggingface models, we need to embed them or intercept the embeddings.
        Alternatively, you can define an approach that modifies the embedding layer directly.
        """
        # The simplest way for sequence models is to let Captum handle the embeddings in the forward pass
        # But you need a model that can accept "embedded" input. Otherwise, you might intercept the model's embedding layer in Captum.
        raise NotImplementedError(
            "Implement or wrap your huggingface model to accept embeddings directly."
        )

    # Alternatively, if your model directly takes input_ids:
    def forward_func_ids(input_ids):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # shape = (batch_size, num_labels)
        return outputs["logits"][:, target_idx]

    ig = IntegratedGradients(forward_func_ids)

    # Baseline is typically zeros
    baseline_input_ids = torch.zeros_like(input_ids).to(input_ids.device)

    # Now compute attributions
    attributions, delta = ig.attribute(
        inputs=input_ids,
        baselines=baseline_input_ids,
        target=0,
        return_convergence_delta=True,
        n_steps=50,
    )

    return attributions, delta


@app.route("/api/integrated-gradients", methods=["POST"])
def integrated_gradients_api():
    try:
        data = request.json
        smiles_list = data.get("smiles_list", [])
        target_idx = data.get("target_idx", 0)

        # Log input data
        logging.info(f"Received SMILES list: {smiles_list}")
        logging.info(f"Target index: {target_idx}")

        # 1) Tokenize
        tokenized = tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        # Log token shapes
        logging.info(f"Input IDs shape: {input_ids.shape}")
        logging.info(f"Attention mask shape: {attention_mask.shape}")

        # 2) Extract knowledge features
        knowledge_feats = []
        for sm in smiles_list:
            knowledge_feats.append(extract_knowledge_features(sm))
        knowledge_feats_tensor = torch.tensor(knowledge_feats, dtype=torch.float).to(
            device
        )

        # Log knowledge features shape
        logging.info(f"Knowledge features shape: {knowledge_feats_tensor.shape}")

        # 3) Compute Integrated Gradients
        attributions, delta = compute_integrated_gradients_embedding(
            knowledge_model=model,
            input_ids=input_ids,
            knowledge_features=knowledge_feats_tensor,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            target_idx=target_idx,
        )

        # Log attributions and delta
        logging.info(f"Attributions shape: {attributions.shape}")
        logging.info(f"Convergence delta: {delta}")

        # 4) Prepare response
        attributions_list = attributions.detach().cpu().numpy().tolist()
        delta_val = float(delta.mean().detach().cpu().item())

        response = {
            "smiles_list": smiles_list,
            "attributions": attributions_list,
            "convergence_delta": delta_val,
        }
        return jsonify(response), 200

    except Exception as e:
        logging.error(f"IG error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import openai

CORS(app)
logging.basicConfig(level=logging.INFO)
openai.api_key = "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"


@app.route("/api/chatgpt", methods=["POST"])
def chatgpt():
    try:
        # Log incoming request
        logging.info("Received request to /api/chatgpt")
        data = request.json
        logging.info(f"Request data: {data}")

        # Extract chat messages and LIME data from request
        messages = data.get("messages", [])
        lime_data = data.get("limeData", [])

        if not messages:
            return jsonify({"error": "No messages provided."}), 400

        # Prepare OpenAI API messages
        chat_messages = [
            {
                "role": "system",
                "content": "You are an assistant that provides insights and answers questions.",
            },
            *[{"role": msg["role"], "content": msg["content"]} for msg in messages],
        ]

        # Include LIME data if available
        if lime_data:
            lime_context = f"The following LIME data provides insights: {lime_data}"
            chat_messages.append({"role": "system", "content": lime_context})

        # Log messages sent to OpenAI API
        logging.info(f"Chat messages: {chat_messages}")

        # Call OpenAI Chat API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=chat_messages,
            temperature=0.7,  # Adjust creativity
        )

        # Extract response content
        ai_response = response.choices[0].message["content"]
        logging.info(f"OpenAI response: {ai_response}")

        return jsonify({"response": ai_response}), 200

    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logging.error(f"Server error: {e}")
        return jsonify({"error": str(e)}), 500


DEEPSEEK_API_KEY = (
    "sk-d65b0704062948acbe7e2d242263c834"  # Replace with your DeepSeek API key
)
DEEPSEEK_API_URL = (
    "https://api.deepseek.com/v1/chat/completions"  # Example DeepSeek API endpoint
)


@app.route("/api/deepseek", methods=["POST"])
def deepseek_endpoint():
    try:
        # Log incoming request
        logging.info("Received request to /api/deepseek")
        data = request.json
        logging.info(f"Request data: {data}")

        # Extract the user's message from the request
        user_message = data.get("message", "")
        if not user_message:
            return jsonify({"error": "No message provided."}), 400

        # Prepare the payload for the DeepSeek API
        payload = {
            "model": "deepseek-chat",  # Replace with the correct model name
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.7,  # Adjust creativity
        }

        # Log payload sent to DeepSeek API
        logging.info(f"Payload sent to DeepSeek API: {payload}")

        # Make the request to the DeepSeek API
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)

        # Check if the request was successful
        if response.status_code != 200:
            logging.error(
                f"DeepSeek API error: {response.status_code} - {response.text}"
            )
            return jsonify({"error": "Failed to get response from DeepSeek API."}), 500

        # Extract the AI's response
        deepseek_response = response.json()
        ai_response = (
            deepseek_response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        logging.info(f"DeepSeek response: {ai_response}")

        # Return the AI's response
        return jsonify({"response": ai_response}), 200

    except Exception as e:
        logging.error(f"Server error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_model_and_tokenizer()  # Load model on startup
    app.run(debug=True)
