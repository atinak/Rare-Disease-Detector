"""
Tests for Predictor Module.
"""

import unittest
import pandas as pd
import numpy as np
import os
from src.predictor import Predictor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from unittest.mock import patch, MagicMock
import joblib


class TestPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create temporary directories
        cls.models_dir = "test_models"
        cls.data_dir = "test_data"
        os.makedirs(cls.models_dir, exist_ok=True)
        os.makedirs(cls.data_dir, exist_ok=True)

        # Create a sample model and feature pipeline
        cls.sample_data = pd.DataFrame({
            'numerical_1': [1, 2, 3, 4, 5],
            'numerical_2': [5, 4, 3, 2, 1],
            'categorical_1': ['A', 'B', 'A', 'C', 'B'],
            'categorical_2': ['X', 'X', 'Y', 'Z', 'Y']
        })
        cls.sample_target = pd.Series([0, 1, 0, 1, 0])

        cls.feature_engineer = FeatureEngineer(models_dir=cls.models_dir)
        cls.X_processed = cls.feature_engineer.create_feature_pipeline(cls.sample_data, cls.sample_target)
        cls.feature_engineer.save_pipeline()
        cls.model_trainer = ModelTrainer(models_dir=cls.models_dir)

        cls.model = cls.model_trainer.train_random_forest(cls.X_processed, cls.sample_target, param_grid = {'n_estimators': [2], 'max_depth':[2]})  # Use small values for speed
        cls.model_trainer.save_model(cls.model, "test_model")
        
        # Initialize Predictor
        cls.predictor = Predictor(model_filename="test_model", models_dir=cls.models_dir)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary directories
         for file in os.listdir(cls.models_dir):
            os.remove(os.path.join(cls.models_dir, file))
         os.rmdir(cls.models_dir)


    def test_initialization(self):
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.feature_pipeline)

    def test_predict(self):
        # Create new data similar to training data
        new_data = pd.DataFrame({
            'numerical_1': [6, 7, 8],
            'numerical_2': [0, 1, 2],
            'categorical_1': ['B', 'C', 'A'],
            'categorical_2': ['Y', 'Z', 'X']
        })
        
        predictions = self.predictor.predict(new_data)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(new_data))  # Check prediction length
        self.assertTrue(all(isinstance(p, (int, np.int64)) for p in predictions)) #check type



    def test_predict_probabilities(self):
        # Create new data
        new_data = pd.DataFrame({
            'numerical_1': [6, 7, 8],
            'numerical_2': [0, 1, 2],
            'categorical_1': ['B', 'C', 'A'],
            'categorical_2': ['Y', 'Z', 'X']
        })

        probabilities = self.predictor.predict(new_data, return_probabilities=True)
        self.assertIsNotNone(probabilities)
        self.assertEqual(len(probabilities), len(new_data)) # Check shape
        self.assertTrue(all(isinstance(p, np.ndarray) for p in probabilities))  # Check return type (array of probabilities)


    @patch('src.predictor.Predictor._load_model')
    @patch('src.predictor.Predictor._load_feature_pipeline')
    def test_predict_failure(self, mock_load_pipeline, mock_load_model):
        # Simulate model/pipeline loading failure
        mock_load_model.return_value = None
        mock_load_pipeline.return_value = None

        predictor = Predictor() # Create a new predictor instance for this test
        new_data = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})  # Dummy data
        result = predictor.predict(new_data)
        self.assertIsNone(result)  # Check if prediction returns None on failure


    def test_transform_features_handles_missing_selected_features(self):
        #This test makes sure that if a selected feature is missing, transform_features correctly creates and fills the missing columns.
        
        #Create a FeatureEngineer instance
        feature_engineer = FeatureEngineer(models_dir=self.models_dir)

        #Sample X and y
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'feature3': [7,8,9]})
        y = pd.Series([0, 1, 0])
        
        #Set selected features manually
        feature_engineer.selected_features = ['feature1', 'feature3', 'missing_feature']
        
        # Mock feature_selector and pca to prevent actual transform
        feature_engineer.feature_selector = MagicMock()
        feature_engineer.pca = MagicMock()
        
        # Create data with missing selected feature
        X_missing = pd.DataFrame({'feature1': [10, 11], 'feature3': [12,13]})
        
        # Transform the data.
        X_transformed = feature_engineer.transform_features(X_missing)
    
        # Check if missing column is present and filled with 0
        self.assertIn('missing_feature', X_transformed.columns)
        self.assertTrue((X_transformed['missing_feature'] == 0).all())