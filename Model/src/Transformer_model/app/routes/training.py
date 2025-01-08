# app/routes/training.py

from flask import Blueprint, request, jsonify
import os
from threading import Thread
import logging

from sklearn.model_selection import train_test_split

from Model.src.Transformer_model.app import ProgressCallback, compute_metrics

from ..utils import load_smiles_data
from ..datasets import KnowledgeAugmentedModel, SMILESDataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import math
import time
import pickle

training_bp = Blueprint("training", __name__)


def train_model_in_background(app, file_path):
    try:
        app.training_status.update(
            {
                "status": "running",
                "message": "Training started.",
                "progress": 0,
                "eta": None,
                "total_steps": None,
            }
        )
        logging.info("Training started.")

        smiles_train, targets_train, knowledge_features_train = load_smiles_data(
            file_path,
            app.rdkit_descriptors,
            properties_present=True,
            fit_scaler=True,
            scalers={
                "scaler_pIC50": app.scaler_pIC50,
                "scaler_logP": app.scaler_logP,
                "scaler_num_atoms": app.scaler_num_atoms,
            },
        )

        (
            smiles_train_split,
            smiles_val,
            targets_train_split,
            targets_val,
            features_train_split,
            features_val,
        ) = train_test_split(
            smiles_train,
            targets_train,
            knowledge_features_train,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )

        train_dataset = SMILESDataset(
            smiles_train_split,
            features_train_split,
            targets_train_split,
            app.tokenizer,
            max_length=128,
        )
        eval_dataset = SMILESDataset(
            smiles_val,
            features_val,
            targets_val,
            app.tokenizer,
            max_length=128,
        )

        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            learning_rate=1e-5,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            gradient_accumulation_steps=1,
            fp16=True,
            max_grad_norm=1.0,
        )

        total_train_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * training_args.world_size
        )
        num_update_steps_per_epoch = math.ceil(
            len(train_dataset) / total_train_batch_size
        )
        total_steps = int(num_update_steps_per_epoch * training_args.num_train_epochs)
        app.training_status["total_steps"] = total_steps

        early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
        trainer = Trainer(
            model=app.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[ProgressCallback(total_steps, app)],
        )

        trainer.train()

        fine_tuned_model_path = app.config["MODEL_PATH"]
        app.model.save_pretrained(fine_tuned_model_path)
        logging.info(f"Fine-tuned model saved to {fine_tuned_model_path}")

        with app.model_lock:
            app.model = KnowledgeAugmentedModel.from_pretrained(
                fine_tuned_model_path, app.rdkit_descriptors_length, num_labels=3
            )
            app.model.to(app.device)
            app.model.eval()

        with open(os.path.join(fine_tuned_model_path, "scalers.pkl"), "wb") as f:
            pickle.dump(
                {
                    "scaler_pIC50": app.scaler_pIC50,
                    "scaler_logP": app.scaler_logP,
                    "scaler_num_atoms": app.scaler_num_atoms,
                },
                f,
            )
        logging.info("Scalers saved successfully.")

        app.training_status.update(
            {
                "status": "completed",
                "message": "Training completed successfully.",
                "progress": 100,
                "eta": "00:00:00",
            }
        )

    except Exception as e:
        app.training_status.update(
            {
                "status": "error",
                "message": str(e),
                "progress": 0,
                "eta": None,
            }
        )
        logging.error(f"Training error: {e}")


@training_bp.route("/train", methods=["POST"])
def train_model_route():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file found in request"}), 400
        file = request.files["file"]
        file_path = os.path.join(
            request.app.config["UPLOAD_FOLDER"],
            f"train_{int(time.time())}_{file.filename}",
        )
        file.save(file_path)
        thread = Thread(target=train_model_in_background, args=(request.app, file_path))
        thread.start()
        return (
            jsonify(
                {
                    "message": "Training started in the background. Check status at /get_training_status."
                }
            ),
            202,
        )
    except Exception as e:
        logging.error(f"Error during training setup: {e}")
        return jsonify({"error": str(e)}), 500


@training_bp.route("/get_training_status", methods=["GET"])
def get_training_status_route():
    return jsonify(request.app.training_status)
