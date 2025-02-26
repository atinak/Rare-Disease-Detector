"""
Feature Engineering Module for Rare Disease Prediction System.

This module handles feature transformation, selection, and encoding for
the rare disease prediction models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict, Union, Optional
import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Class for advanced feature engineering operations on Orphanet data.
    
    This class handles feature scaling, selection, dimensionality reduction,
    and encoding for improving model performance.
    """
    
    def __init__(self, models_dir: str = "../models"):
        """
        Initialize the FeatureEngineer.
        
        Args:
            models_dir: Directory where preprocessing models will be stored
        """
        self.models_dir = models_dir
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.encoders = {}
        self.numerical_columns = []
        self.categorical_columns = []
        self.selected_features = []
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
    
    def identify_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify numerical and categorical features in the dataset.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Tuple of (numerical column names, categorical column names)
        """
        logger.info("Identifying feature types")
        
        numerical_cols = []
        categorical_cols = []
        
        for col in X.columns:
            # Check if the column has numeric data
            if X[col].dtype in ['int64', 'float64']:
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
        
        self.numerical_columns = numerical_cols
        self.categorical_columns = categorical_cols
        
        logger.info(f"Identified {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features")
        return numerical_cols, categorical_cols
    
    def handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values")
        
        # For HPO features (numerical), fill NaN with 0 (absence of symptom)
        # For categorical features, fill with most frequent value
        
        # Identify feature types if not already done
        if not self.numerical_columns and not self.categorical_columns:
            self.identify_feature_types(X)
        
        # Handle numerical features
        X_num = X[self.numerical_columns].fillna(0)
        
        # Handle categorical features
        X_cat = X[self.categorical_columns].copy()
        if not X_cat.empty:
            for col in X_cat.columns:
                X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0] if not X_cat[col].mode().empty else "UNKNOWN")
        
        # Combine processed features
        X_processed = pd.concat([X_num, X_cat], axis=1)
        
        logger.info(f"Handled missing values in {X.shape[1]} features")
        return X_processed


    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit the scaler on this data
            
        Returns:
            DataFrame with scaled features
        """
        logger.info("Scaling numerical features")
        
        # Identify feature types if not already done
        if not self.numerical_columns and not self.categorical_columns:
            self.identify_feature_types(X)
        
        # Get numerical columns
        X_num = X[self.numerical_columns].copy()
        
        # Create and fit scaler if needed
        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            X_num_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_num),
                columns=X_num.columns,
                index=X_num.index
            )
        else:
            X_num_scaled = pd.DataFrame(
                self.scaler.transform(X_num),
                columns=X_num.columns,
                index=X_num.index
            )
        
        # Combine with categorical features
        X_cat = X[self.categorical_columns].copy() if self.categorical_columns else pd.DataFrame(index=X.index)
        X_scaled = pd.concat([X_num_scaled, X_cat], axis=1)
        
        logger.info(f"Scaled {len(self.numerical_columns)} numerical features")
        return X_scaled
    
    def encode_categorical_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using one-hot encoding.
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit the encoders on this data
            
        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features")
        
        # Identify feature types if not already done
        if not self.numerical_columns and not self.categorical_columns:
            self.identify_feature_types(X)
        
        # No categorical features to encode
        if not self.categorical_columns:
            logger.info("No categorical features to encode")
            return X
        
        # Get numerical features
        X_num = X[self.numerical_columns].copy()
        
        # Initialize list to store encoded DataFrames
        encoded_dfs = [X_num]
        
        # Encode each categorical feature
        for col in self.categorical_columns:
            # Handle potential NaN values
            X[col] = X[col].fillna('UNKNOWN')
            
            # Create encoder if needed
            if fit or col not in self.encoders:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_data = encoder.fit_transform(X[[col]])
                self.encoders[col] = encoder
            else:
                encoder = self.encoders[col]
                encoded_data = encoder.transform(X[[col]])
            
            # Create DataFrame with encoded data
            encoded_df = pd.DataFrame(
                encoded_data,
                columns=[f"{col}_{val}" for val in encoder.categories_[0]],
                index=X.index
            )
            
            # Add to list of encoded DataFrames
            encoded_dfs.append(encoded_df)
        
        # Combine all encoded features
        X_encoded = pd.concat(encoded_dfs, axis=1)
        
        logger.info(f"Encoded {len(self.categorical_columns)} categorical features")
        return X_encoded
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 100, method: str = 'mi', fit: bool = True) -> pd.DataFrame:
        """
        Select top features using statistical methods.

        Args:
            X: Feature DataFrame
            y: Target variable
            n_features: Number of features to select
            method: Feature selection method ('mi' for mutual info, 'chi2' for chi-squared)
            fit: Whether to fit the selector on this data

        Returns:
            DataFrame with selected features
        """
        logger.info(f"Selecting top {n_features} features using {method}")

        # Adjust n_features if it exceeds available features
        n_features = min(n_features, X.shape[1])

    def select_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 100, method: str = 'mi', fit: bool = True) -> pd.DataFrame:
        """
        Select top features using statistical methods.

        Args:
            X: Feature DataFrame
            y: Target variable
            n_features: Number of features to select
            method: Feature selection method ('mi' for mutual info, 'chi2' for chi-squared)
            fit: Whether to fit the selector on this data

        Returns:
            DataFrame with selected features
        """
        logger.info(f"Selecting top {n_features} features using {method}")

        # Adjust n_features if it exceeds available features
        n_features = min(n_features, X.shape[1])

        # --- EMPTY DATAFRAME CHECK ---
        if X.empty or X.shape[0] == 0 or X.shape[1] == 0:  # Check for empty DataFrame OR if 0 columns
            logger.warning("Empty DataFrame passed to select_features or DataFrame with 0 columns. Returning empty DataFrame with n_features columns.")
            return pd.DataFrame(columns=[f"feature_{i}" for i in range(n_features)])  # Return empty DataFrame
        # -----------------------------


        # Create feature selector if needed
        if fit or self.feature_selector is None:
            if method == 'mi':
                 # --- FORCE CONTINUOUS FEATURES ---
                self.feature_selector = SelectKBest(lambda X, y: mutual_info_classif(X, y, discrete_features=False), k=n_features)
                # ------------------------------------
            elif method == 'chi2':
                # Ensure all features are non-negative for chi2
                X_non_neg = X.copy()
                for col in X.columns:
                    if X_non_neg[col].min() < 0:
                        X_non_neg[col] = X_non_neg[col] - X_non_neg[col].min()
                self.feature_selector = SelectKBest(chi2, k=n_features)
                X = X_non_neg  # Use the non-negative version for chi2
            else:
                raise ValueError(f"Unknown feature selection method: {method}")

            # Fit and transform
            # Check if y has samples. If not, return empty DataFrame with appropriate columns.
            if len(y) == 0:
                return pd.DataFrame(columns=[f"feature_{i}" for i in range(n_features)]) # Return DF with correct number of columns
            X_selected = self.feature_selector.fit_transform(X, y)

            # Get selected feature names
            selected_mask = self.feature_selector.get_support()
            self.selected_features = X.columns[selected_mask].tolist()
        else:
            # Transform using existing selector
            X_selected = self.feature_selector.transform(X)

        # Convert to DataFrame
        X_selected_df = pd.DataFrame(
            X_selected,
            columns=self.selected_features,
            index=X.index  # Keep the original index
        )

        logger.info(f"Selected {len(self.selected_features)} features")
        return X_selected_df
    
    def reduce_dimensions(self, X: pd.DataFrame, n_components: int = 50, fit: bool = True) -> pd.DataFrame:
        """
        Reduce dimensions using PCA.

        Args:
            X: Feature DataFrame
            n_components: Number of PCA components
            fit: Whether to fit PCA on this data

        Returns:
            DataFrame with reduced dimensions
        """
        logger.info(f"Reducing dimensions to {n_components} components")

        # --- ADDED EMPTY/SINGLE-COLUMN CHECK ---
        if X.empty or X.shape[1] <= 1:  # Check for empty or single-column DataFrame
            logger.warning("Empty DataFrame or only one column in reduce_dimensions. Returning original DataFrame (no PCA).")
            return X.copy()  # Return a copy to avoid modifying the original
        # ---------------------------------------

        # Adjust n_components if it exceeds available features
        n_components = min(n_components, X.shape[1])

        # Create and fit PCA if needed
        if fit or self.pca is None:
            self.pca = PCA(n_components=n_components)
            X_pca = self.pca.fit_transform(X)
        else:
            X_pca = self.pca.transform(X)

        # Convert to DataFrame
        X_pca_df = pd.DataFrame(
            X_pca,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=X.index
        )

        # Log explained variance
        if fit or self.pca is None:
            explained_var = self.pca.explained_variance_ratio_.sum()
            logger.info(f"PCA explains {explained_var:.2%} of variance with {n_components} components")

        return X_pca_df
    
    def create_feature_pipeline(self, X: pd.DataFrame, y: pd.Series, use_pca: bool = True,
                              n_features: int = 100, n_components: int = 50) -> pd.DataFrame:
        """
        Apply the full feature engineering pipeline.

        Args:
            X: Raw feature DataFrame
            y: Target variable
            use_pca: Whether to use PCA dimensionality reduction
            n_features: Number of features to select
            n_components: Number of PCA components (if use_pca is True)

        Returns:
            Fully processed feature DataFrame
        """
        logger.info("Applying full feature engineering pipeline")

        # Identify feature types
        self.identify_feature_types(X)

        # Handle missing values
        X_processed = self.handle_missing_values(X)

        # Scale features
        X_scaled = self.scale_features(X_processed, fit=True)

        # Encode categorical features
        X_encoded = self.encode_categorical_features(X_scaled, fit=True)

        # --- ADDED EMPTY DATAFRAME CHECK AFTER ENCODING ---
        if X_encoded.shape[0] == 0:
            logger.warning("Empty DataFrame after encoding. Returning empty DataFrame.")
            return pd.DataFrame()  # Or return X_encoded, since it's already empty
        # ----------------------------------------------------

        # Select features
        X_selected = self.select_features(X_encoded, y, n_features=n_features, fit=True)

        # Reduce dimensions if requested
        if use_pca:
            X_final = self.reduce_dimensions(X_selected, n_components=n_components, fit=True)
        else:
            X_final = X_selected

        logger.info(f"Feature pipeline complete: {X.shape} -> {X_final.shape}")
        return X_final
    
    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the existing pipeline.
        
        Args:
            X: Raw feature DataFrame
            
        Returns:
            Processed feature DataFrame
        """
        if self.scaler is None:
            raise ValueError("Feature pipeline not initialized. Call create_feature_pipeline() first.")
        
        # Handle missing values
        X_processed = self.handle_missing_values(X)
        
        # Scale features
        X_scaled = self.scale_features(X_processed, fit=False)
        
        # Encode categorical features
        X_encoded = self.encode_categorical_features(X_scaled, fit=False)
        
        # Select features
        if self.feature_selector is not None:
            # Check if all selected features are present
            missing_cols = set(self.selected_features) - set(X_encoded.columns)
            for col in missing_cols:
                X_encoded[col] = 0  # Add missing columns with default values
            
            # Select only the columns needed for the feature selector
            X_selected = X_encoded[self.selected_features]
        else:
            X_selected = X_encoded
        
        # Reduce dimensions if PCA was used
        if self.pca is not None:
            X_final = self.reduce_dimensions(X_selected, fit=False)
        else:
            X_final = X_selected
        
        return X_final
    
    def save_pipeline(self, filename: str = "feature_pipeline") -> None:
        """
        Save the feature engineering pipeline objects.
        
        Args:
            filename: Base filename for saved objects
        """
        logger.info(f"Saving feature engineering pipeline to {self.models_dir}")
        
        # Create dictionary of objects to save
        pipeline_objects = {
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'pca': self.pca,
            'encoders': self.encoders,
            'selected_features': self.selected_features,
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns
        }
        
        # Save each object individually
        for key, value in pipeline_objects.items():
            joblib.dump(value, os.path.join(self.models_dir, f"{filename}_{key}.joblib"))

        logger.info("Feature engineering pipeline saved successfully.")


    def load_pipeline(self, filename: str = "feature_pipeline") -> None:
        """
        Load the feature engineering pipeline objects.

        Args:
            filename: Base filename for saved objects.
        """
        logger.info(f"Loading feature engineering pipeline from {self.models_dir}")

        try:
            self.scaler = joblib.load(os.path.join(self.models_dir, f"{filename}_scaler.joblib"))
            self.feature_selector = joblib.load(os.path.join(self.models_dir, f"{filename}_feature_selector.joblib"))
            self.pca = joblib.load(os.path.join(self.models_dir, f"{filename}_pca.joblib"))
            self.encoders = joblib.load(os.path.join(self.models_dir, f"{filename}_encoders.joblib"))
            self.selected_features = joblib.load(os.path.join(self.models_dir, f"{filename}_selected_features.joblib"))
            self.numerical_columns = joblib.load(os.path.join(self.models_dir, f"{filename}_numerical_columns.joblib"))
            self.categorical_columns = joblib.load(os.path.join(self.models_dir, f"{filename}_categorical_columns.joblib"))
            logger.info("Feature engineering pipeline loaded successfully.")

        except FileNotFoundError as e:
            logger.error(f"Error loading pipeline: {e}.  Make sure the pipeline has been saved.")
            raise  # Re-raise the exception to halt execution if critical components are missing.
