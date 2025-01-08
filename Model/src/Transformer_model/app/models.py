# app/models.py

import os
import logging
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch

from .datasets import KnowledgeAugmentedModel


def load_model_and_tokenizer(app, num_labels=3):
    try:
        app.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        logging.info("Tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        raise e

    model_path = app.config["MODEL_PATH"]
    knowledge_dim = 3 + app.rdkit_descriptors_length  # Define this appropriately
    if os.path.exists(model_path) and os.path.isdir(model_path):
        try:
            logging.info("Attempting to load the fine-tuned model.")
            model_local = KnowledgeAugmentedModel.from_pretrained(
                model_path, knowledge_dim, num_labels
            )
            model_local.to(app.device)
            model_local.eval()
            app.model = model_local
            logging.info("Fine-tuned model loaded successfully.")

            # Deserialize scalers from disk
            scalers_path = os.path.join(model_path, "scalers.pkl")
            if os.path.exists(scalers_path):
                with open(scalers_path, "rb") as f:
                    scalers = pickle.load(f)
                    app.scaler_pIC50 = scalers["scaler_pIC50"]
                    app.scaler_logP = scalers["scaler_logP"]
                    app.scaler_num_atoms = scalers["scaler_num_atoms"]
                logging.info("Scalers loaded successfully.")
            else:
                logging.error("Scalers file not found. Predictions may fail.")
        except Exception as e:
            logging.error(
                f"Failed to load fine-tuned model: {e}. Falling back to base model."
            )
            base_model = AutoModelForSequenceClassification.from_pretrained(
                "seyonec/ChemBERTa-zinc-base-v1",
                num_labels=num_labels,
                problem_type="regression",
            )
            model_local = KnowledgeAugmentedModel(
                base_model, knowledge_dim, num_labels=num_labels
            )
            model_local.to(app.device)
            model_local.eval()
            app.model = model_local
    else:
        logging.info(
            "Fine-tuned model directory not found. Loading base pre-trained model."
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "seyonec/ChemBERTa-zinc-base-v1",
            num_labels=num_labels,
            problem_type="regression",
        )
        model_local = KnowledgeAugmentedModel(
            base_model, knowledge_dim, num_labels=num_labels
        )
        model_local.to(app.device)
        model_local.eval()
        app.model = model_local


def initialize_model(app):
    app.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.model = None
    app.tokenizer = None
    app.scaler_pIC50 = None
    app.scaler_logP = None
    app.scaler_num_atoms = None
    # Define rdkit_descriptors_length based on your descriptors
    from rdkit.Chem import Descriptors

    app.rdkit_descriptors_length = len(Descriptors._descList)
    load_model_and_tokenizer(app)
