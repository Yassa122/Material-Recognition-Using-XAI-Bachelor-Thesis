# app/config.py

import os


class Config:
    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    RESULTS_FOLDER = os.path.join(os.getcwd(), "results")
    LOGS_FOLDER = os.path.join(os.getcwd(), "logs")
    MODEL_PATH = os.path.join(os.getcwd(), "fine_tuned_chemberta_with_knowledge")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    ALLOWED_EXTENSIONS = {"csv"}
