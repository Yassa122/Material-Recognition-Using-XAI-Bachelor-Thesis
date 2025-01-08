# app/__init__.py

import logging
import os
from flask import Flask
from flask_cors import CORS
from threading import Lock

# Import blueprints
from .routes.training import training_bp
# from .routes.prediction import prediction_bp
from .routes.explanations import explanations_bp
from .routes.async_tasks import async_tasks_bp


def create_app():
    app = Flask(__name__)

    # Load configurations
    app.config.from_object("app.config.Config")

    # Enable CORS
    CORS(app)

    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler("logs/app.log"), logging.StreamHandler()],
    )

    # Ensure necessary directories exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)
    os.makedirs(app.config["LOGS_FOLDER"], exist_ok=True)

    # Register blueprints
    app.register_blueprint(training_bp)
    # app.register_blueprint(prediction_bp)
    app.register_blueprint(explanations_bp)
    app.register_blueprint(async_tasks_bp)

    # Initialize global locks
    app.model_lock = Lock()
    app.shap_lock = Lock()

    # Load model and tokenizer
    from .models import initialize_model

    initialize_model(app)

    return app
