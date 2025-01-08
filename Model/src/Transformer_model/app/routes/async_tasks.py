# app/routes/async_tasks.py

from flask import Blueprint, request, jsonify, send_file
from threading import Thread
import os
import uuid
import logging
import pandas as pd
from ..utils import calculate_descriptors  # Define this in utils.py

async_tasks_bp = Blueprint("async_tasks", __name__)
tasks = {}


def process_calculation(app, task_id, file_path):
    try:
        logging.info(f"Task {task_id}: Starting calculation.")
        app.tasks[task_id]["status"] = "running"
        df = pd.read_csv(file_path)
        if "SMILES" not in df.columns:
            app.tasks[task_id]["status"] = "error"
            app.tasks[task_id]["message"] = "CSV must contain a 'SMILES' column."
            logging.error(f"Task {task_id}: 'SMILES' column missing.")
            return

        rdkit_descriptors = app.rdkit_descriptors_length  # Adjust accordingly
        descriptors = df["SMILES"].apply(
            lambda sm: calculate_descriptors(sm, rdkit_descriptors)
        )
        descriptors_df = pd.DataFrame(descriptors.tolist())
        combined_df = pd.concat([df, descriptors_df], axis=1)

        invalid_rows = combined_df[combined_df["Valid_SMILES"] == False]
        if not invalid_rows.empty:
            logging.warning(f"Task {task_id}: Some SMILES strings could not be parsed.")

        output_filename = f"calculated_properties_{task_id}.csv"
        output_path = os.path.join(app.config["RESULTS_FOLDER"], output_filename)
        os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)
        combined_df.to_csv(output_path, index=False)

        app.tasks[task_id]["status"] = "completed"
        app.tasks[task_id]["result"] = output_filename
        app.tasks[task_id]["message"] = "Calculation completed successfully."
        logging.info(f"Task {task_id}: Calculation completed.")

    except Exception as e:
        app.tasks[task_id]["status"] = "error"
        app.tasks[task_id]["message"] = str(e)
        logging.error(f"Task {task_id}: Error during calculation - {e}")


@async_tasks_bp.route("/calculate_properties_async", methods=["POST"])
def calculate_properties_async_route():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected for uploading"}), 400
        if file and request.app.config["ALLOWED_EXTENSIONS"]:
            task_id = str(uuid.uuid4())
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            upload_filename = f"{task_id}_{file.filename}"
            upload_path = os.path.join(
                request.app.config["UPLOAD_FOLDER"], upload_filename
            )
            os.makedirs(request.app.config["UPLOAD_FOLDER"], exist_ok=True)
            file.save(upload_path)
            request.app.tasks[task_id] = {
                "status": "pending",
                "message": "Task is pending and will start shortly.",
                "result": None,
                "uploaded_at": timestamp,
            }
            thread = Thread(
                target=process_calculation, args=(request.app, task_id, upload_path)
            )
            thread.start()
            logging.info(f"Task {task_id}: Received and started.")
            return jsonify({"task_id": task_id}), 202
        else:
            return jsonify({"error": "Allowed file types are csv"}), 400
    except Exception as e:
        logging.error(f"Error in /calculate_properties_async: {e}")
        return jsonify({"error": str(e)}), 500


@async_tasks_bp.route("/calculate_properties_status/<task_id>", methods=["GET"])
def calculate_properties_status_route(task_id):
    try:
        if task_id not in request.app.tasks:
            return jsonify({"error": "Invalid task ID"}), 404
        task = request.app.tasks[task_id]
        response = {
            "task_id": task_id,
            "status": task["status"],
            "message": task["message"],
        }
        if task["status"] == "completed":
            response["download_url"] = f"/download_results/{task['result']}"
        elif task["status"] == "error":
            response["error_detail"] = task["message"]
        return jsonify(response), 200
    except Exception as e:
        logging.error(f"Error in /calculate_properties_status/{task_id}: {e}")
        return jsonify({"error": str(e)}), 500


@async_tasks_bp.route("/download_results/<filename>", methods=["GET"])
def download_results_route(filename):
    try:
        file_path = os.path.join(request.app.config["RESULTS_FOLDER"], filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        return send_file(
            file_path, mimetype="text/csv", as_attachment=True, download_name=filename
        )
    except Exception as e:
        logging.error(f"Error in /download_results/{filename}: {e}")
        return jsonify({"error": str(e)}), 500
