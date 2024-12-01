from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from tqdm import tqdm  # Import the tqdm function
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
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


# Status of the training
training_status = {"status": "idle", "message": ""}


# Background thread for training the model
def train_model_in_background(file_path):
    global training_status
    try:
        # Set status to 'running'
        training_status["status"] = "running"
        training_status["message"] = "Training in progress..."
        logging.info("Training started.")

        # Load training data
        smiles_train, targets_train, knowledge_features_train = load_smiles_data(
            file_path, properties_present=True
        )

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
        logging.info("Training completed.")

    except Exception as e:
        # Update status to 'error'
        training_status["status"] = "error"
        training_status["message"] = str(e)
        logging.error(f"Training error: {e}")


def explain_shap_as_json(file_path):
    try:
        # Load predictions
        predictions_file_path = file_path.replace(
            ".csv", "shap_1732931106_predicted_properties.csv.csv"
        )
        predictions_df = pd.read_csv(predictions_file_path)
        predictions = predictions_df[
            ["Predicted_pIC50", "Predicted_logP", "Predicted_num_atoms"]
        ].values

        # Reduce sample size for SHAP
        sampled_predictions = predictions[:100]  # Use only the first 100 samples

        # SHAP explanation
        def model_predict(inputs):
            inputs = torch.tensor(inputs, dtype=torch.float).to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids=None, attention_mask=None, knowledge_features=inputs
                )
                return outputs["logits"].cpu().numpy()

        # Use k-means to summarize the background data
        explainer = shap.KernelExplainer(
            model_predict, shap.kmeans(sampled_predictions, 10)
        )
        shap_values = explainer.shap_values(sampled_predictions)

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


# Route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    if training_status["status"] == "running":
        return jsonify({"error": "Model is being trained. Try again later."}), 400

    try:
        file = request.files["file"]
        file_path = os.path.join(
            UPLOAD_FOLDER, f"predict_{int(time.time())}_{file.filename}"
        )
        file.save(file_path)

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
            predictions_df.to_csv(
                file_path.replace(".csv", "_predictions.csv"), index=False
            )

        prediction_thread = Thread(target=run_predictions)
        prediction_thread.start()

        return jsonify({"message": "Prediction started in the background."}), 202

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    try:
        # Get pagination parameters from the query string
        page = int(request.args.get("page", 1))  # Default to page 1 if not provided
        limit = int(request.args.get("limit", 5))  # Default to 5 records per page

        predictions_file_path = os.path.join(UPLOAD_FOLDER, "predicted_properties.csv")

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


# SHAP Explanation Processing Function
# Updated SHAP Explanation Function
# SHAP Explanation Processing Function


def explain_shap_in_background(file_path):
    global training_status
    try:
        training_status["status"] = "running"
        training_status["message"] = "Processing SHAP explanation..."
        logging.info("SHAP explanation started.")

        # Check if predictions already exist
        predictions_file_path = file_path.replace(".csv", "_predicted_properties.csv")

        # Log the upload folder path
        logging.info(f"Checking UPLOAD_FOLDER: {UPLOAD_FOLDER}")

        if not os.path.exists(predictions_file_path):
            # If predictions do not exist, perform predictions first
            smiles_predict, _, knowledge_features_predict = load_smiles_data(
                file_path, properties_present=False
            )

            # Prepare the dataset for prediction
            predict_dataset = SMILESDataset(
                smiles_predict,
                knowledge_features_predict,
                tokenizer=tokenizer,
                max_length=128,
            )
            dataloader = DataLoader(predict_dataset, batch_size=16)

            # Prepare the model in evaluation mode
            model.eval()

            # Collect predictions for SHAP explanation
            predictions = []
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Predicting for SHAP explanation"):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    logits = outputs["logits"]
                    predictions.extend(logits.cpu().numpy())

            # Convert the predictions to a numpy array
            predictions = np.array(predictions)

            # Save predictions to CSV
            predictions_df = pd.DataFrame(
                predictions,
                columns=["Predicted_pIC50", "Predicted_logP", "Predicted_num_atoms"],
            )
            predictions_df["SMILES"] = smiles_predict
            predictions_df.to_csv(predictions_file_path, index=False)

            # Log successful save and check folder contents
            logging.info(f"Predictions saved to {predictions_file_path}")

            # Check if the predictions file is in the UPLOAD_FOLDER
            if os.path.exists(predictions_file_path):
                logging.info(
                    f"Predictions file successfully saved at: {predictions_file_path}"
                )
            else:
                logging.error(f"Predictions file not found at: {predictions_file_path}")

            # Log contents of the UPLOAD_FOLDER to verify if the file exists
            logging.info(f"Current files in UPLOAD_FOLDER: {os.listdir(UPLOAD_FOLDER)}")

        # If predictions exist, load them
        predictions_df = pd.read_csv(predictions_file_path)
        predictions = predictions_df[
            ["Predicted_pIC50", "Predicted_logP", "Predicted_num_atoms"]
        ].values

        # Define a function to make predictions for SHAP explainer
        def model_predict(inputs):
            inputs = torch.tensor(inputs, dtype=torch.float).to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids=None, attention_mask=None, knowledge_features=inputs
                )
                return outputs["logits"].cpu().numpy()

        # Initialize SHAP KernelExplainer or DeepExplainer
        explainer = shap.KernelExplainer(model_predict, predictions)
        shap_values = explainer.shap_values(predictions)

        # Process SHAP values and save the explanations
        shap.summary_plot(shap_values, predictions)

        # Once done, update the status
        training_status["status"] = "completed"
        training_status["message"] = "SHAP explanation completed successfully."
        logging.info("SHAP explanation completed.")
    except Exception as e:
        training_status["status"] = "error"
        training_status["message"] = f"Error during SHAP explanation: {str(e)}"
        logging.error(f"SHAP explanation error: {str(e)}")


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


def run_predictions(file_path):
    global prediction_status
    try:
        # Set status to 'running'
        prediction_status["status"] = "running"
        prediction_status["message"] = "Prediction in progress..."

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

        # Update status to 'completed'
        prediction_status["status"] = "completed"
        prediction_status["message"] = (
            f"Prediction completed successfully. Results saved to {predictions_file}."
        )

    except Exception as e:
        # Update status to 'error'
        prediction_status["status"] = "error"
        prediction_status["message"] = str(e)


@app.route("/get_prediction_status", methods=["GET"])
def get_prediction_status():
    return jsonify(prediction_status)


if __name__ == "__main__":
    load_model_and_tokenizer()  # Load model on startup
    app.run(debug=True)
