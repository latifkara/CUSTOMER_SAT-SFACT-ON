from tkinter import N
import pandas as pd
from zenml import step
import logging
import mlflow
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import numpy as np
from src.TrainTestSplitData import TrainTestSplitData
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

class TrainModel:
    """
    Train multiple models on preprocessed data
    Args:
        split: TrainTestSplitData containing preprocessed train/test data
    Returns:
        Dictionary of trained models
    """
    def __init__(self, split: TrainTestSplitData):
        self.split = split
        self.models: Dict[str, Any] = {}
        self.importance_features: Dict[str, List[str]] = {}
        
        # Initialize classifiers
        self.classifiers = [
            ('DT', DecisionTreeClassifier(random_state=42)),
            ('RF', RandomForestClassifier(random_state=42, n_estimators=100)),
            ('XGBoost', XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss', 
                random_state=42,
                verbosity=0
            )),
            ('LightGBM', LGBMClassifier(
                random_state=42, 
                verbose=-1,
                objective='binary',
                force_col_wise=True
            ))
        ]

    def ensure_numeric_data(self, X: pd.DataFrame) -> np.ndarray:
        """
        Ensure data is in proper numeric format for training
        """
        try:
            # Convert to numpy array if it's a DataFrame
            if isinstance(X, pd.DataFrame):
                # Check for any remaining non-numeric columns
                non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
                if len(non_numeric_cols) > 0:
                    logging.warning(f"Found non-numeric columns after preprocessing: {list(non_numeric_cols)}")
                    # Convert to numeric where possible, errors='coerce' will turn invalid values to NaN
                    for col in non_numeric_cols:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                    
                    # Fill any NaN values created during conversion
                    X = X.fillna(0)
                
                return X.values
            else:
                # Already a numpy array
                return X
        except Exception as e:
            logging.error(f"Error in ensure_numeric_data: {str(e)}")
            # Fallback: try to convert directly
            return np.array(X, dtype=np.float32)

    def plot_importance(self, importance, feature_names, num=10, save=False):
        """
        Plot and return top important features
        Args:
            importance: Feature importance values
            feature_names: List of feature names
            num: Number of top features to return
            save: Whether to save the plot
        Returns:
            List of top feature names
        """
        try:
            # Ensure we have feature names
            if feature_names is None or len(feature_names) == 0:
                feature_names = [f'feature_{i}' for i in range(len(importance))]
            
            # Ensure we don't request more features than available
            num = min(num, len(importance), len(feature_names))
            
            feature_imp = pd.DataFrame({
                'Value': importance[:len(feature_names)], 
                'Feature': feature_names[:len(importance)]
            })
            
            # Sort by importance
            feature_imp = feature_imp.sort_values('Value', ascending=False)
            
            if save:
                plt.figure(figsize=(10, 8))
                sns.barplot(data=feature_imp.head(num), x='Value', y='Feature')
                plt.title('Feature Importance')
                plt.tight_layout()
                plt.savefig('importances.png')
                plt.close()
            
            # Return list of feature names
            return feature_imp['Feature'].head(num).tolist()
            
        except Exception as e:
            logging.error(f"Error in plot_importance: {str(e)}")
            # Return first num features if importance plotting fails
            return feature_names[:num] if feature_names else []

    def create_fresh_classifier(self, name: str):
        """
        Create a fresh instance of a classifier
        """
        if name == 'DT':
            return DecisionTreeClassifier(random_state=42)
        elif name == 'RF':
            return RandomForestClassifier(random_state=42, n_estimators=100)
        elif name == 'XGBoost':
            return XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss', 
                random_state=42,
                verbosity=0
            )
        elif name == 'LightGBM':
            return LGBMClassifier(
                random_state=42, 
                verbose=-1,
                objective='binary',
                force_col_wise=True
            )
        else:
            raise ValueError(f"Unknown classifier name: {name}")

    def train_model(self) -> Dict[str, Any]:
        """
        Train all models on preprocessed data
        Returns:
            Dictionary of trained models with metadata
        """
        try:
            logging.info("Starting model training on preprocessed data...")
            logging.info(f"Training data shape: {self.split.X_train.shape}")
            logging.info(f"Test data shape: {self.split.X_test.shape}")
            logging.info(f"Target variable shape: {self.split.y_train.shape}")
            
            # Ensure data is in proper format for training
            X_train_processed = self.ensure_numeric_data(self.split.X_train)
            X_test_processed = self.ensure_numeric_data(self.split.X_test)
            
            # Ensure y_train is in proper format (1D array)
            y_train = np.array(self.split.y_train).ravel()
            
            logging.info(f"After conversion - X_train shape: {X_train_processed.shape}")
            logging.info(f"After conversion - y_train shape: {y_train.shape}")
            
            # Get feature names
            if hasattr(self.split.X_train, 'columns'):
                feature_names = list(self.split.X_train.columns)
            else:
                feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]
            
            logging.info(f"Number of features: {len(feature_names)}")
            
            # Step 1: Train initial models to get feature importance
            logging.info("Step 1: Training models for feature importance...")
            initial_models = {}

            for name, classifier in self.classifiers:
                logging.info(f"Training initial {name} model...")
                try:
                    # Disable autolog temporarily to avoid the Series issue
                    mlflow.sklearn.autolog(disable=True)
                    
                    model = classifier.fit(X_train_processed, y_train)
                    initial_models[name] = model
                    logging.info(f"{name} trained successfully")
                    
                except Exception as e:
                    logging.error(f"Error training {name}: {str(e)}")
                    continue
            
            if not initial_models:
                raise ValueError("No models were successfully trained in the initial phase")
            
            # Step 2: Extract feature importance
            logging.info("Step 2: Extracting feature importance...")
            
            for name, model in initial_models.items():
                try:
                    if hasattr(model, 'feature_importances_'):
                        # Get indices of top features
                        feature_importance_values = model.feature_importances_
                        top_feature_names = self.plot_importance(
                            feature_importance_values, 
                            feature_names, 
                            15
                        )
                        self.importance_features[name] = top_feature_names
                        logging.info(f"{name} - Selected {len(top_feature_names)} top features")
                    else:
                        # Fallback: use all features
                        self.importance_features[name] = feature_names
                        logging.warning(f"{name} doesn't have feature_importances_, using all features")
                        
                except Exception as e:
                    logging.error(f"Error extracting importance for {name}: {str(e)}")
                    # Fallback: use all features
                    self.importance_features[name] = feature_names
            
            # Step 3: Retrain models with selected features
            logging.info("Step 3: Retraining models with selected features...")
            
            for name, _ in self.classifiers:
                if name in self.importance_features:
                    try:
                        selected_feature_names = self.importance_features[name]
                        logging.info(f"Retraining {name} with {len(selected_feature_names)} features...")
                        
                        # Get indices of selected features
                        selected_indices = []
                        for fname in selected_feature_names:
                            try:
                                idx = feature_names.index(fname)
                                selected_indices.append(idx)
                            except ValueError:
                                logging.warning(f"Feature {fname} not found in feature names")
                                continue
                        
                        if not selected_indices:
                            logging.warning(f"No valid feature indices found for {name}, using all features")
                            X_train_selected = X_train_processed
                        else:
                            X_train_selected = X_train_processed[:, selected_indices]
                        
                        # Create new instance of classifier for retraining
                        new_classifier = self.create_fresh_classifier(name)
                        
                        # Train with selected features
                        trained_model = new_classifier.fit(X_train_selected, y_train)
                        
                        # Store the trained model
                        self.models[name] = trained_model
                        logging.info(f"{name} retrained successfully with selected features")
                        
                    except Exception as e:
                        logging.error(f"Error retraining {name}: {str(e)}")
                        # Fallback: use the initial model with all features
                        if name in initial_models:
                            self.models[name] = initial_models[name]
                            # Also store all features as selected features for this model
                            self.importance_features[name] = feature_names
                            logging.warning(f"Using initial {name} model as fallback")
                else:
                    logging.warning(f"No feature importance found for {name}, skipping retraining")
            
            if not self.models:
                raise ValueError("No models were successfully trained or retrained")
            
            logging.info(f"Model training completed. Trained {len(self.models)} models.")
            logging.info(f"Successfully trained models: {list(self.models.keys())}")
            
            return {
                'models': self.models,
                'feature_importance': self.importance_features,
                'feature_names': feature_names,
                'processed_data': {
                    'X_train': X_train_processed,
                    'X_test': X_test_processed
                }
            }
            
        except Exception as e:
            logging.error(f"Error in train_model: {str(e)}")
            raise

    def get_model_performance(self, X_train_processed: np.ndarray, y_train: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """
        Evaluate model performance on training data
        Returns:
            Dictionary of model accuracies
        """
        performance = {}
        
        for name, model in self.models.items():
            try:
                selected_feature_names = self.importance_features[name]
                
                # Get indices of selected features
                selected_indices = []
                for fname in selected_feature_names:
                    try:
                        idx = feature_names.index(fname)
                        selected_indices.append(idx)
                    except ValueError:
                        continue
                
                if selected_indices:
                    X_selected = X_train_processed[:, selected_indices]
                else:
                    X_selected = X_train_processed
                
                y_pred = model.predict(X_selected)
                accuracy = accuracy_score(y_train, y_pred)
                performance[name] = accuracy
                
                logging.info(f"{name} training accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logging.error(f"Error evaluating {name}: {str(e)}")
                performance[name] = 0.0
        
        return performance


@step(experiment_tracker=experiment_tracker.name)
def train_model(split: TrainTestSplitData) -> Dict[str, Any]:
    """
    Train multiple models on preprocessed data
    Args:
        split: TrainTestSplitData containing preprocessed X_train, X_test, y_train, y_test
    Returns:
        Dictionary containing trained models and performance metrics
    """
    try:
        logging.info("Initializing TrainModel for preprocessed data...")
        trainer = TrainModel(split)
        
        # Train the models
        result = trainer.train_model()
        
        # Ensure y_train is in proper format for performance evaluation
        y_train = np.array(split.y_train).ravel()
        
        # Log performance
        performance = trainer.get_model_performance(
            result['processed_data']['X_train'], 
            y_train,
            result['feature_names']
        )
        logging.info("Model Performance Summary:")
        for name, acc in performance.items():
            logging.info(f"  {name}: {acc:.4f}")
        
        # Add performance to result
        result['performance'] = performance
        
        # Debug logging
        logging.info(f"Final result keys: {list(result.keys())}")
        logging.info(f"Models in result: {list(result['models'].keys())}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error in train_model step: {str(e)}")
        raise