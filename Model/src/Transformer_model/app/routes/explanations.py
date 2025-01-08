# app/routes/explanations.py

from flask import Blueprint, request, jsonify
from threading import Thread
import os
import logging
from ..explanations.shap_explainer import explain_shap_as_json

# Import other explainers similarly

explanations_bp = Blueprint("explanations", __name__)


def background_shap_computation(app, file_path):
    try:
        app.shap_status.update(
            {"status": "running", "message": "SHAP explanation is being processed..."}
        )
        shap_result = explain_shap_as_json(app, file_path)
        app.shap_status.update(
            {
                "status": "completed",
                "result": shap_result,
                "message": "SHAP explanation completed successfully.",
            }
        )
    except Exception as e:
        app.shap_status.update(
            {"status": "error", "message": f"Error during SHAP explanation: {str(e)}"}
        )
        logging.error(f"SHAP explanation error: {e}")


@explanations_bp.route("/start_shap_explanation", methods=["POST"])
def start_shap_explanation():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file found in request"}), 400
        file = request.files["file"]
        file_path = os.path.join(
            request.app.config["UPLOAD_FOLDER"],
            f"shap_{int(time.time())}_{file.filename}",
        )
        file.save(file_path)
        thread = Thread(
            target=background_shap_computation, args=(request.app, file_path)
        )
        thread.start()
        return (
            jsonify({"message": "SHAP explanation started.", "status": "running"}),
            202,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@explanations_bp.route("/shap_status", methods=["GET"])
def get_shap_status():
    return jsonify(request.app.shap_status)


# Similarly, add routes for LIME, IG, and Attention explanations
