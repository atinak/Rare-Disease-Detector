"""
Tests for Feature Engineering Module.
"""

import unittest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer  # Correct import
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from unittest.mock import patch
import os
import joblib

class TestFeatureEngineer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for saving models
        cls.models_dir = "test_models"
        os.makedirs(cls.models_dir, exist_ok=True)
        cls.feature_engineer = FeatureEngineer(models_dir=cls.models_dir)

        # Create a sample DataFrame for testing
        cls.sample_data = pd.DataFrame({
            'numerical_1': [1, 2, 3, np.nan, 5],
            'numerical_2': [5, 4, 3, 2, 1],
            'categorical_1': ['A', 'B', 'A', 'C', 'B'],
            'categorical_2': ['X', 'X', 'Y', 'Z', 'Y']
        })
        cls.sample_target = pd.Series([0, 1, 0, 1, 0])

    @classmethod
    def tearDownClass(cls):
        # Remove the temporary directory
        for file in os.listdir(cls.models_dir):
            os.remove(os.path.join(cls.models_dir, file))
        os.rmdir(cls.models_dir)
    
    def test_identify_feature_types(self):
        num_cols, cat_cols = self.feature_engineer.identify_feature_types(self.sample_data)
        self.assertEqual(num_cols, ['numerical_1', 'numerical_2'])
        self.assertEqual(cat_cols, ['categorical_1', 'categorical_2'])
        self.assertEqual(self.feature_engineer.numerical_columns, ['numerical_1', 'numerical_2'])
        self.assertEqual(self.feature_engineer.categorical_columns, ['categorical_1', 'categorical_2'])

    def test_handle_missing_values(self):
      X_processed = self.feature_engineer.handle_missing_values(self.sample_data)
      self.assertFalse(X_processed.isnull().any().any())  # Check no missing values
      self.assertEqual(X_processed['numerical_1'][3], 0)  # Check numerical fill
      self.assertEqual(X_processed['categorical_1'][3], 'A')  # Check categorical fill (most frequent)


    def test_scale_features(self):
        # Fit scaler
        X_scaled_fit = self.feature_engineer.scale_features(self.sample_data, fit=True)
        self.assertIsNotNone(self.feature_engineer.scaler)
        self.assertTrue(np.allclose(X_scaled_fit[['numerical_1', 'numerical_2']].mean(), [0, 0], atol=1e-6))
        self.assertTrue(np.allclose(X_scaled_fit[['numerical_1', 'numerical_2']].std(), [1, 1], atol=1e-6))


        # Transform using fitted scaler
        X_scaled_trans = self.feature_engineer.scale_features(self.sample_data, fit=False)
        self.assertTrue(np.allclose(X_scaled_fit, X_scaled_trans)) # Check consistency between fit and transform


    def test_encode_categorical_features(self):
        # Fit encoders
        X_encoded_fit = self.feature_engineer.encode_categorical_features(self.sample_data, fit=True)
        self.assertEqual(len(self.feature_engineer.encoders), 2)  # Check number of encoders
        self.assertIn('categorical_1_A', X_encoded_fit.columns)  # Check encoded column names
        self.assertIn('categorical_2_Z', X_encoded_fit.columns)

        # Transform using fitted encoders
        X_encoded_trans = self.feature_engineer.encode_categorical_features(self.sample_data, fit=False)
        self.assertTrue(np.allclose(X_encoded_fit, X_encoded_trans))  #Check consistency


    def test_select_features(self):
        X_processed = self.feature_engineer.handle_missing_values(self.sample_data)
        X_encoded = self.feature_engineer.encode_categorical_features(X_processed)
        # Fit selector (mutual info)
        X_selected_fit = self.feature_engineer.select_features(X_encoded, self.sample_target, n_features=3, method='mi', fit=True)
        self.assertEqual(len(self.feature_engineer.selected_features), 3)
        self.assertEqual(X_selected_fit.shape[1], 3)

        # Transform using fitted selector
        X_selected_trans = self.feature_engineer.select_features(X_encoded, self.sample_target, n_features=3, method='mi', fit=False)
        self.assertTrue(np.array_equal(X_selected_fit, X_selected_trans))  # Check consistency

    def test_reduce_dimensions(self):
        X_processed = self.feature_engineer.handle_missing_values(self.sample_data)

        # Fit PCA
        X_pca_fit = self.feature_engineer.reduce_dimensions(X_processed, n_components=2, fit=True)
        self.assertIsNotNone(self.feature_engineer.pca)
        self.assertEqual(X_pca_fit.shape[1], 2)

        # Transform using fitted PCA
        X_pca_trans = self.feature_engineer.reduce_dimensions(X_processed, n_components=2, fit=False)
        self.assertTrue(np.allclose(X_pca_fit, X_pca_trans, atol=1e-6))  # Check consistency.  PCA can have sign flips.


    def test_create_feature_pipeline(self):
        X_final = self.feature_engineer.create_feature_pipeline(self.sample_data, self.sample_target, use_pca=True, n_features=3, n_components=2)
        self.assertIsNotNone(self.feature_engineer.scaler)
        self.assertIsNotNone(self.feature_engineer.feature_selector)
        self.assertIsNotNone(self.feature_engineer.pca)
        self.assertEqual(X_final.shape[1], 2) #check shape
        self.assertTrue((X_final.dtypes != 'object').all()) # Check if all columns are numeric
        self.assertTrue(X_final.notna().all().all())  #check no NaN

    def test_transform_features(self):
      # First, create the pipeline
        self.feature_engineer.create_feature_pipeline(self.sample_data, self.sample_target)

        # Now, transform new data
        new_data = pd.DataFrame({
            'numerical_1': [6, 7, np.nan],
            'numerical_2': [0, 1, 2],
            'categorical_1': ['B', 'C', 'A'],
            'categorical_2': ['Y', 'Z', 'X']
        })
        X_transformed = self.feature_engineer.transform_features(new_data)
        self.assertEqual(X_transformed.shape[1], len(self.feature_engineer.selected_features))  # Check shape after transform
        self.assertTrue((X_transformed.dtypes != 'object').all()) # Check if all columns are numeric
        self.assertTrue(X_transformed.notna().all().all())  #check no NaN


    def test_save_and_load_pipeline(self):
        # Create the pipeline
        self.feature_engineer.create_feature_pipeline(self.sample_data, self.sample_target)
        # Save the pipeline
        self.feature_engineer.save_pipeline("test_pipeline")

        # Create a new FeatureEngineer instance
        loaded_engineer = FeatureEngineer(models_dir=self.models_dir)
        # Load the pipeline
        loaded_engineer.load_pipeline("test_pipeline")

        # Check if all components are loaded correctly
        self.assertIsNotNone(loaded_engineer.scaler)
        self.assertIsNotNone(loaded_engineer.feature_selector)
        self.assertIsNotNone(loaded_engineer.pca)
        self.assertGreater(len(loaded_engineer.encoders), 0)  # Check for encoders
        self.assertGreater(len(loaded_engineer.selected_features), 0)
        self.assertGreater(len(loaded_engineer.numerical_columns), 0)
        self.assertGreater(len(loaded_engineer.categorical_columns), 0)

        # Test transform_features with loaded pipeline
        new_data = pd.DataFrame({
            'numerical_1': [6, 7, np.nan],
            'numerical_2': [0, 1, 2],
            'categorical_1': ['B', 'C', 'A'],
            'categorical_2': ['Y', 'Z', 'X']
            })

        X_transformed = self.feature_engineer.transform_features(new_data.copy())
        X_transformed_loaded = loaded_engineer.transform_features(new_data.copy())
        self.assertTrue(np.allclose(X_transformed, X_transformed_loaded, atol=1e-06)) # Check that transform gives same result