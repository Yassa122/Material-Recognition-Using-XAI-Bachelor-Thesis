# app/explanations/shap_explainer.py

import shap
import numpy as np
import pandas as pd
import logging
from transformers import AutoTokenizer
import torch


def explain_shap_as_json(app, file_path):
    try:
        predictions_file_path = app.config["UPLOAD_FOLDER"] + "/predictions.csv"
        predictions_df = pd.read_csv(predictions_file_path)
        predictions = predictions_df[
            ["Predicted_pIC50", "Predicted_logP", "Predicted_num_atoms"]
        ].values

        sampled_size = min(len(predictions), 100)
        sampled_indices = np.arange(sampled_size)

        with app.shap_lock:
            if len(app.global_smiles_predict) == 0:
                raise ValueError("No prediction data available for SHAP explanation.")

            if (
                len(app.global_input_ids_list) == 0
                or len(app.global_attention_mask_list) == 0
            ):
                raise ValueError(
                    "Prediction inputs (input_ids or attention_mask) are missing."
                )

            sampled_indices = sampled_indices[
                sampled_indices < len(app.global_smiles_predict)
            ]

            knowledge_features_np = np.array(app.global_knowledge_features_predict)[
                sampled_indices
            ]
            input_ids_np = app.global_input_ids_list[sampled_indices]
            attention_mask_np = app.global_attention_mask_list[sampled_indices]
            smiles_sub = [app.global_smiles_predict[i] for i in sampled_indices]

        def model_predict(features_batch):
            try:
                model_cpu = app.model.to("cpu")
                model_cpu.eval()
                features_tensor = torch.tensor(features_batch, dtype=torch.float).to(
                    "cpu"
                )
                batch_size = features_tensor.shape[0]
                input_ids_tensor = app.tokenizer(
                    smiles_sub,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                )["input_ids"].repeat(batch_size, 1)
                attention_mask_tensor = app.tokenizer(
                    smiles_sub,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                )["attention_mask"].repeat(batch_size, 1)

                with torch.no_grad():
                    outputs = model_cpu(
                        input_ids=input_ids_tensor,
                        attention_mask=attention_mask_tensor,
                        knowledge_features=features_tensor,
                    )
                    return outputs["logits_pIC50"].numpy()
            except Exception as e:
                logging.error(f"Error in model_predict during SHAP explanation: {e}")
                raise e

        background = shap.kmeans(knowledge_features_np, 10)
        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(knowledge_features_np)

        response = {
            "features": ["Predicted_pIC50", "Predicted_logP", "Predicted_num_atoms"],
            "shap_values": shap_values.tolist(),
            "base_value": explainer.expected_value,
        }
        return response

    except Exception as e:
        logging.error(f"SHAP explanation error: {str(e)}")
        return {"error": str(e)}
