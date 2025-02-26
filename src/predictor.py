"""
Prediction Module for Rare Disease Prediction System.

This module handles loading trained models and making predictions on new data.
"""

import pandas as pd
from typing import Dict, Any, Optional
import logging
import os
from .feature_engineering import FeatureEngineer  # Corrected import
from .model_training import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Predictor:
    """
    Class for making predictions using a trained model.

    This class loads a pre-trained model and feature engineering pipeline,
    and provides a method to predict rare diseases based on input features.
    """

    def __init__(self, model_filename: str = "best_random_forest", feature_pipeline_filename: str = "feature_pipeline", models_dir: str = "../models"):
        """
        Initialize the Predictor.

        Args:
            model_filename: Filename of the saved model.
            feature_pipeline_filename: Filename of the saved feature pipeline.
        """
        self.models_dir = models_dir
        self.model_filename = model_filename
        self.feature_pipeline_filename = feature_pipeline_filename
        self.model = self._load_model()
        self.feature_pipeline = self._load_feature_pipeline()

    def _load_model(self) -> Any:
        """
        Load the trained model.

        Returns:
            Loaded model, or None if loading fails.
        """
        model_trainer = ModelTrainer(models_dir=self.models_dir)
        return model_trainer.load_model(self.model_filename)

    def _load_feature_pipeline(self) -> Any:
        """
        Load the feature engineering pipeline.

        Returns:
            Loaded feature pipeline, or None if loading fails.
        """
        feature_engineer = FeatureEngineer(models_dir=self.models_dir)
        try:
            # Assuming FeatureEngineer has a load_pipeline method
            feature_engineer.scaler = joblib.load(os.path.join(self.models_dir, self.feature_pipeline_filename + "_scaler.joblib"))
            feature_engineer.feature_selector = joblib.load(os.path.join(self.models_dir, self.feature_pipeline_filename + "_feature_selector.joblib"))
            feature_engineer.pca = joblib.load(os.path.join(self.models_dir, self.feature_pipeline_filename + "_pca.joblib"))
            # Load encoders
            encoders_path = os.path.join(self.models_dir, self.feature_pipeline_filename + "_encoders.joblib")
            if os.path.exists(encoders_path):
                feature_engineer.encoders = joblib.load(encoders_path)

            selected_features_path = os.path.join(self.models_dir, self.feature_pipeline_filename + "_selected_features.joblib")
            if os.path.exists(selected_features_path):
              feature_engineer.selected_features = joblib.load(selected_features_path)

            # Load column types
            numerical_columns_path = os.path.join(self.models_dir, self.feature_pipeline_filename + "_numerical_columns.joblib")
            if os.path.exists(numerical_columns_path):
                feature_engineer.numerical_columns = joblib.load(numerical_columns_path)
            
            categorical_columns_path = os.path.join(self.models_dir, self.feature_pipeline_filename + "_categorical_columns.joblib")
            if os.path.exists(categorical_columns_path):
                feature_engineer.categorical_columns = joblib.load(categorical_columns_path)

            return feature_engineer
        except FileNotFoundError:
            logger.error(f"Feature pipeline file not found: {self.feature_pipeline_filename}")
            return None

    def predict(self, input_data: pd.DataFrame, return_probabilities: bool = False) -> Any:
        """
        Make predictions on new data.

        Args:
            input_data: Input data as a DataFrame.
            return_probabilities: Whether to return class probabilities.

        Returns:
            Predicted OrphaCode(s) or class probabilities.  Returns None if
            prediction fails.
        """
        if self.model is None or self.feature_pipeline is None:
            logger.error("Model or feature pipeline not loaded.")
            return None

        try:
            # Transform features
            processed_data = self.feature_pipeline.transform_features(input_data)

            # Make predictions
            if return_probabilities:
                try:
                    predictions = self.model.predict_proba(processed_data)
                except AttributeError:
                    logger.error("Model does not support probability prediction.")
                    return None
            else:
                predictions = self.model.predict(processed_data)

            return predictions

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return None