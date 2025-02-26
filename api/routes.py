"""
API Routes for Rare Disease Prediction System.

This module defines the API endpoints and their corresponding logic.
"""

from flask import render_template, request, jsonify
from api.app import app  # Import the app instance
from src.predictor import Predictor
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the predictor (load the model and feature pipeline)
predictor = Predictor()
hpo_mapping = {}  # This will be populated later if needed

@app.route('/', methods=['GET'])
def index():
    """
    Render the home page.
    """
    return render_template('index.html', hpo_mapping=hpo_mapping)  # Pass the hpo_mapping


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    """
    try:
        data = request.get_json()  # Get the JSON data from the request
        logger.info(f"Received prediction request: {data}")
        # Ensure data is in DataFrame format
        input_df = pd.DataFrame([data]) 

        # Make predictions using the Predictor class
        predictions = predictor.predict(input_df)

        if predictions is not None:
            # Convert predictions to a list for JSON serialization
            result = {'predictions': predictions.tolist()}
            return jsonify(result)  # Return the prediction result as JSON
        else:
            return jsonify({'error': 'Prediction failed'}), 500 # Internal Server Error


    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        return jsonify({'error': str(e)}), 400  # Bad Request