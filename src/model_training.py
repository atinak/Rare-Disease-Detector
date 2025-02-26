"""
Model Training Module for Rare Disease Prediction System.

This module handles model training, hyperparameter optimization, and saving 
of trained models for the rare disease prediction system.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from typing import Dict, Tuple, Any, Union, Optional
import logging
import xgboost as xgb
import optuna

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Class for training and evaluating machine learning models.

    This class handles model initialization, training, hyperparameter tuning,
    cross-validation, and saving the best model.
    """

    def __init__(self, models_dir: str = "../models"):
        """
        Initialize the ModelTrainer.

        Args:
            models_dir: Directory to save trained models.
        """
        self.models_dir = models_dir
        self.best_model = None
        self.best_params = None
        self.model_performance = {}
        os.makedirs(models_dir, exist_ok=True)  # Ensure models directory exists


    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                             param_grid: Optional[Dict[str, Any]] = None,
                             cv_folds: int = 5) -> RandomForestClassifier:
        """
        Train a Random Forest model with hyperparameter tuning.

        Args:
            X_train: Training features.
            y_train: Training target.
            param_grid: Hyperparameter grid for tuning.
            cv_folds: Number of cross-validation folds.

        Returns:
            Trained Random Forest model.
        """
        logger.info("Training Random Forest model")

        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

        rf_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf_model, param_grid, cv=cv_folds, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        logger.info(f"Best Random Forest parameters: {self.best_params}")
        return self.best_model


    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      param_grid: Optional[Dict[str, Any]] = None,
                      cv_folds: int = 5) -> xgb.XGBClassifier:
        """
        Train an XGBoost model with hyperparameter tuning.

        Args:
            X_train: Training features.
            y_train: Training target.
            param_grid: Hyperparameter grid for tuning.
            cv_folds: Number of cross-validation folds.

        Returns:
            Trained XGBoost model.
        """
        logger.info("Training XGBoost model")

        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
            }
            
        xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(y_train.unique()), random_state=42, use_label_encoder=False)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=cv_folds, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        logger.info(f"Best XGBoost parameters: {self.best_params}")
        return self.best_model

    def train_with_optuna(self, X_train: pd.DataFrame, y_train: pd.Series, model_type: str = 'random_forest', n_trials: int = 20):
      """
      Train a model with hyperparameter optimization using Optuna.

      Args:
          X_train (pd.DataFrame): Training features.
          y_train (pd.Series): Training target.
          model_type (str): Type of model ('random_forest' or 'xgboost').
          n_trials (int): Number of trials for Optuna optimization.

      Returns:
          Any: Best trained model.
      """
      def objective(trial):
          if model_type == 'random_forest':
              param = {
                  'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                  'max_depth': trial.suggest_int('max_depth', 5, 30),
                  'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                  'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                  'random_state': 42,
              }
              model = RandomForestClassifier(**param)
          elif model_type == 'xgboost':
              param = {
                  'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                  'max_depth': trial.suggest_int('max_depth', 3, 10),
                  'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                  'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                  'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                  'random_state': 42,
                  'objective': 'multi:softmax',
                  'use_label_encoder': False,
              }
              model = xgb.XGBClassifier(**param)
          else:
              raise ValueError("Invalid model_type. Choose 'random_forest' or 'xgboost'.")

          model.fit(X_train, y_train)  # Fit the model here
          y_pred = model.predict(X_train)
          f1 = f1_score(y_train, y_pred, average='weighted')
          return f1

      study = optuna.create_study(direction='maximize')
      study.optimize(objective, n_trials=n_trials)

      best_trial = study.best_trial
      logger.info(f"Best trial for {model_type}: {best_trial.params}")
      
      if model_type == 'random_forest':
          self.best_model = RandomForestClassifier(**best_trial.params, random_state=42)
      elif model_type == 'xgboost':
          best_trial.params['objective'] = 'multi:softmax'
          best_trial.params['use_label_encoder'] = False
          self.best_model = xgb.XGBClassifier(**best_trial.params, random_state=42)
      
      self.best_model.fit(X_train, y_train) #refit with best params
      self.best_params = best_trial.params
      return self.best_model
    
    def run_training_pipeline(self, X: pd.DataFrame, y: pd.Series, model_selection:str = 'random_forest', use_optuna: bool = True) -> Tuple[Any, Dict[str, Any]]:

        logger.info("Running model training pipeline")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert y_train and y_test to string type
        y_train = y_train.astype(str)
        y_test = y_test.astype(str)

        # Select and train model
        if model_selection == 'random_forest':
            if use_optuna:
                model = self.train_with_optuna(X_train, y_train, model_type='random_forest')
            else:
                model = self.train_random_forest(X_train, y_train)
            self.evaluate_model(model, X_test, y_test, model_name='Random Forest')
        elif model_selection == 'xgboost':
            if use_optuna:
                model = self.train_with_optuna(X_train, y_train, model_type='xgboost')
            else: 
                model = self.train_xgboost(X_train, y_train)
            self.evaluate_model(model, X_test, y_test, model_name='XGBoost')
        else:
            raise ValueError("Invalid model_selection. Choose 'random_forest' or 'xgboost'.")

        self.save_model(model, filename=f'best_{model_selection}')  # Save the trained model

        return model, self.best_params
    

    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name:str) -> None:
        """
        Evaluate the trained model.

        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test target.
        """
        logger.info("Evaluating model")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        try:
            y_pred_proba = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')  # One-vs-Rest
            logger.info(f"AUC: {auc:.4f}")
            self.model_performance[model_name] = {'accuracy': accuracy, 'f1': f1, 'auc': auc}

        except AttributeError:
            logger.warning("AUC cannot be computed (predict_proba not available)")
            self.model_performance[model_name] = {'accuracy': accuracy, 'f1': f1}


        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score (weighted): {f1:.4f}")
        logger.info("Classification Report:\n" + classification_report(y_test, y_pred))



    def save_model(self, model: Any, filename: str) -> None:
        """
        Save the trained model.

        Args:
            model: Trained model to save.
            filename: Filename for the saved model.
        """
        filepath = os.path.join(self.models_dir, filename + ".joblib")
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")


    def load_model(self, filename: str) -> Any:
        """
        Load a trained model.

        Args:
            filename: Filename of the saved model.

        Returns:
            Loaded model.
        """
        filepath = os.path.join(self.models_dir, filename + ".joblib")
        if os.path.exists(filepath):
            model = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
            return model
        else:
            logger.error(f"Model file not found: {filepath}")
            return None

