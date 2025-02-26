"""
Flask API for Rare Disease Prediction System.

This module sets up the Flask application and imports the routes.
"""

from flask import Flask

app = Flask(__name__)

# Import the routes (must be done after creating the app object)
from api import routes

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)  # Run on all interfaces, port 5000