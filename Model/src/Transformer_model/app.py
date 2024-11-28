from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
from torch import nn
from threading import Thread

# Flask App Initialization
app = Flask(__name__)
CORS(app)

# Global Variables
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Function to extract knowledge-based features
def extract_knowledge_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        molecular_weight = Descriptors.MolWt(mol)
        num_h_donors = Descriptors.NumHDonors(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        return [molecular_weight, num_h_donors, num_h_acceptors]
    else:
        return [0, 0, 0]


# Load SMILES Data
def load_smiles_data(csv_file, properties_present=True):
    data = pd.read_csv(csv_file)
    smiles = data["SMILES"].tolist()
    knowledge_features = [extract_knowledge_features(sm) for sm in smiles]

    if properties_present:
        targets = data.drop(columns=["SMILES", "mol"], errors="ignore")
        targets = targets.apply(pd.to_numeric, errors="coerce").dropna()
        valid_indices = targets.index
        smiles = [smiles[i] for i in valid_indices]
        knowledge_features = [knowledge_features[i] for i in valid_indices]
        targets = targets.values
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


# Load Pre-trained Model and Tokenizer
def load_model_and_tokenizer():
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "seyonec/ChemBERTa-zinc-base-v1", num_labels=3, problem_type="regression"
    )
    knowledge_dim = 3
    model = KnowledgeAugmentedModel(base_model, knowledge_dim, num_labels=3)
    model.to(device)


training_status = {"status": "idle", "message": ""}

def train_model_in_background(file_path):
    global training_status
    try:
        # Set status to 'running'
        training_status["status"] = "running"
        training_status["message"] = "Training in progress..."

        # Load training data
        smiles_train, targets_train, knowledge_features_train = load_smiles_data(
            file_path, properties_present=True
        )

        # Split into training and validation sets
        from sklearn.model_selection import train_test_split
        smiles_train, smiles_val, targets_train, targets_val, features_train, features_val = train_test_split(
            smiles_train, targets_train, knowledge_features_train, test_size=0.2, random_state=42
        )

        # Prepare datasets
        train_dataset = SMILESDataset(
            smiles_train, features_train, targets_train, tokenizer, max_length=128
        )
        eval_dataset = SMILESDataset(
            smiles_val, features_val, targets_val, tokenizer, max_length=128
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Train the model
        trainer.train()

        # Save the fine-tuned model
        model_path = "./fine_tuned_chemberta_with_knowledge"
        trainer.save_model(model_path)

        # Update status to 'completed'
        training_status["status"] = "completed"
        training_status["message"] = "Training completed successfully. Model saved."

    except Exception as e:
        # Update status to 'error'
        training_status["status"] = "error"
        training_status["message"] = str(e)


@app.route("/train", methods=["POST"])
def train_model():
    try:
        # Save the uploaded file
        if "file" not in request.files:
            return jsonify({"error": "No file found in request"}), 400
        file = request.files["file"]
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Start background training
        thread = Thread(target=train_model_in_background, args=(file_path,))
        thread.start()

        # Respond immediately
        return jsonify({"message": "Training started in the background. Check status at /status."}), 202

    except Exception as e:
        print(f"Error during training setup: {e}")
        return jsonify({"error": str(e)}), 500

    # Endpoint to get the current training status


@app.route("/get_training_status", methods=["GET"])
def get_training_status():
    return jsonify(training_status)


from threading import Thread
from flask import jsonify


@app.route("/predict", methods=["POST"])
def predict():
    # Get the uploaded file
    file = request.files["file"]

    # Save the file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Run the prediction task in a separate thread
    def run_predictions():
        smiles_predict, _, knowledge_features_predict = load_smiles_data(
            file_path, properties_present=False
        )
        predict_dataset = SMILESDataset(
            smiles_predict,
            knowledge_features_predict,
            tokenizer=tokenizer,
            max_length=128,
        )
        dataloader = DataLoader(predict_dataset, batch_size=16)

        predictions = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs["logits"]
                predictions.extend(logits.cpu().numpy())

        # Save predictions to CSV
        predictions_df = pd.DataFrame(
            predictions,
            columns=["Predicted_pIC50", "Predicted_logP", "Predicted_num_atoms"],
        )
        predictions_df["SMILES"] = smiles_predict
        predictions_file = os.path.join(UPLOAD_FOLDER, "predictions.csv")
        predictions_df.to_csv(predictions_file, index=False)

    # Start the prediction task in a new thread
    thread = Thread(target=run_predictions)
    thread.start()

    # Return a response immediately after file upload
    return jsonify(
        {
            "message": "File uploaded successfully. Predictions are being processed in the background."
        }
    )


if __name__ == "__main__":
    load_model_and_tokenizer()
    app.run(debug=True, host="0.0.0.0", port=5000)
