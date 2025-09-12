import pandas as pd
from zenml import step
import logging
import mlflow
from sklearn.ensemble import VotingClassifier
import numpy as np
from src.TrainTestSplitData import TrainTestSplitData
from zenml.client import Client
from mlflow.models import infer_signature
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature

experiment_tracker = Client().active_stack.experiment_tracker

class TrainVottingClassifier:
    """
    Train a voting classifier with preprocessed data
    Args:
        train_out: Dictionary containing trained models and preprocessing info
        split: TrainTestSplitData object containing training data
    Returns:
        A voting classifier
    """
    def __init__(self, train_out: dict, split: TrainTestSplitData):
        self.train_out = train_out
        self.split = split
        
        # Extract components from train_out
        if 'models' in train_out:
            self.models = train_out['models']
            self.feature_importance = train_out.get('feature_importance', {})
            self.preprocessors = train_out.get('preprocessors', {})
            self.processed_feature_names = train_out.get('processed_feature_names', [])
            self.processed_data = train_out.get('processed_data', {})
        else:
            # Fallback if structure is different
            self.models = train_out
            self.feature_importance = {}
            self.preprocessors = {}
            self.processed_feature_names = []
            self.processed_data = {}

    def train_voting_classifier(self):
        """
        Train a voting classifier with error handling for preprocessed data
        """
        try:
            print("Model Keys:", list(self.models.keys()))
            logging.info(f"Available models: {list(self.models.keys())}")
            logging.info(f"importance Features: {self.feature_importance}")
            
            # Check which models are available
            available_models = []
            model_mapping = {
                'DT': 'Decision Tree',
                'RF': 'Random Forest', 
                'XGBoost': 'XGBoost',
                'LightGBM': 'LightGBM'
            }
            
            for key, name in model_mapping.items():
                if key in self.models:
                    available_models.append((key, self.models[key]))
                    logging.info(f"{name} model found")
                else:
                    logging.warning(f"{name} model not found in trained models")
            
            if len(available_models) < 2:
                raise ValueError(f"Need at least 2 models for voting classifier, but only {len(available_models)} available")
            
            # Prepare training data
            if 'X_train' in self.processed_data:
                # Use preprocessed data
                X_train = self.processed_data['X_train']
                logging.info("Using preprocessed training data from train_out")
                logging.info(f"Preprocessed training data shape: {X_train.shape}")
            else:
                # Need to preprocess the data ourselves
                logging.info("Preprocessing data for voting classifier...")
                if 'main' in self.preprocessors:
                    # Use the stored preprocessor
                    preprocessor = self.preprocessors['main']
                    X_train = preprocessor.transform(self.split.X_train)
                elif 'label_encoders' in self.preprocessors:
                    # Use label encoders
                    X_train_copy = self.split.X_train.copy()
                    label_encoders = self.preprocessors['label_encoders']
                    
                    for col, encoder in label_encoders.items():
                        if col in X_train_copy.columns:
                            X_train_copy[col] = X_train_copy[col].fillna('Unknown')
                            X_train_copy[col] = encoder.transform(X_train_copy[col])
                    
                    # Handle numerical columns
                    numerical_cols = X_train_copy.select_dtypes(include=[np.number]).columns
                    for col in numerical_cols:
                        X_train_copy[col] = X_train_copy[col].fillna(X_train_copy[col].median())
                    
                    X_train = X_train_copy.values
                else:
                    raise ValueError("No preprocessor available and no preprocessed data provided")
            
            # For voting classifier, we'll use all features rather than individual feature selections
            # This ensures all models receive the same input format
            logging.info(f"Training voting classifier with data shape: {X_train.shape}")
            
            # Create estimators list
            estimators = []
            for key, model in available_models:
                estimators.append((key, model))
                logging.info(f"Added {model_mapping[key]} to voting classifier")
            
            # Create Voting Classifier
            voting_clf = VotingClassifier(
                estimators=estimators,
                voting='hard'  # Change to 'soft' if you want probability-based voting
            )
            
            # Train the voting classifier
            # Note: We need to retrain the models with the full feature set for consistency
            try:
                voting_clf.fit(X_train, self.split.y_train)
                logging.info(f"Voting classifier trained successfully with {len(estimators)} models")

                # Create MLflow signature and input example
                self._log_model_with_signature(voting_clf, X_train)
                
                logging.info("Model Saved successfully with signature and input example")
                
            except Exception as e:
                logging.error(f"Error training voting classifier: {str(e)}")
                # Try alternative approach: create new voting classifier with retrained models
                logging.info("Attempting alternative approach with fresh model training...")
                voting_clf = self.create_fresh_voting_classifier(X_train)
            
            return voting_clf
            
        except Exception as e:
            logging.error(f"Error training voting classifier: {str(e)}")
            raise

    def _log_model_with_signature(self, model, X_train):
        """
        Log model to MLflow with proper signature and input example
        """
        try:
            # Get real feature names from various sources
            feature_names = self._get_real_feature_names(X_train)
            
            # Create input example with proper feature names
            if isinstance(X_train, pd.DataFrame):
                input_example = X_train.head(1)
                logging.info(f"Using DataFrame columns: {input_example.columns.tolist()}")
            elif isinstance(X_train, np.ndarray):
                # Convert numpy array to DataFrame with real feature names
                input_example = pd.DataFrame(X_train[:1], columns=feature_names)
                logging.info(f"Created DataFrame from numpy array with columns: {feature_names}")
            else:
                raise ValueError(f"Unsupported data type for X_train: {type(X_train)}")
            
            # Make a prediction to infer output schema
            sample_prediction = model.predict(input_example.values if isinstance(input_example, pd.DataFrame) else input_example)
            
            # Create signature using the DataFrame with proper column names
            signature = infer_signature(input_example, sample_prediction)
            
            # Log model with signature and input example
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name="voting_classifier_model"
            )
            
            logging.info("Model logged successfully with signature and input example")
            logging.info(f"Feature names used: {input_example.columns.tolist()}")
            logging.info(f"Model signature: {signature}")
            
        except Exception as e:
            logging.error(f"Error logging model with signature: {str(e)}")
            # Fallback: log without signature
            mlflow.sklearn.log_model(model, artifact_path="model")
            logging.warning("Model logged without signature due to error")

    def _get_real_feature_names(self, X_train):
        """
        Get real feature names from various sources in order of preference
        """
        # Option 1: If X_train is already a DataFrame, use its columns
        if isinstance(X_train, pd.DataFrame):
            return X_train.columns.tolist()
        
        # Option 2: Use processed_feature_names if available
        if self.processed_feature_names and len(self.processed_feature_names) > 0:
            logging.info(f"Using processed_feature_names: {self.processed_feature_names[:5]}...")
            return self.processed_feature_names
        
        # Option 3: Try to get from original split data
        if hasattr(self.split, 'X_train') and isinstance(self.split.X_train, pd.DataFrame):
            original_features = self.split.X_train.columns.tolist()
            logging.info(f"Using original X_train columns: {original_features[:5]}...")
            return original_features
        
        # Option 4: Try to reconstruct from train_out metadata
        if 'feature_names' in self.train_out:
            logging.info(f"Using feature_names from train_out: {self.train_out['feature_names'][:5]}...")
            return self.train_out['feature_names']
        
        # Option 5: Check if we have column info in preprocessors
        if 'column_names' in self.preprocessors:
            logging.info(f"Using column_names from preprocessors: {self.preprocessors['column_names'][:5]}...")
            return self.preprocessors['column_names']
        
        # Option 6: Last resort - create descriptive names based on data shape
        n_features = X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])
        feature_names = [f"feature_{i}" for i in range(n_features)]
        logging.warning(f"Using generic feature names: {feature_names[:5]}... (showing first 5)")
        return feature_names

    def create_fresh_voting_classifier(self, X_train):
        """
        Create a fresh voting classifier by retraining models with consistent data
        """
        try:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier
            from xgboost import XGBClassifier
            from lightgbm import LGBMClassifier
            
            # Create fresh classifiers
            fresh_classifiers = []
            
            if 'DT' in self.models:
                dt = DecisionTreeClassifier(random_state=42)
                dt.fit(X_train, self.split.y_train)
                fresh_classifiers.append(('DT', dt))
                logging.info("Fresh DT model trained")
            
            if 'RF' in self.models:
                rf = RandomForestClassifier(random_state=42, n_estimators=100)
                rf.fit(X_train, self.split.y_train)
                fresh_classifiers.append(('RF', rf))
                logging.info("Fresh RF model trained")
            
            if 'XGBoost' in self.models:
                xgb = XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    verbosity=0
                )
                xgb.fit(X_train, self.split.y_train)
                fresh_classifiers.append(('XGBoost', xgb))
                logging.info("Fresh XGBoost model trained")
            
            if 'LightGBM' in self.models:
                lgb = LGBMClassifier(random_state=42, verbose=-1)
                lgb.fit(X_train, self.split.y_train)
                fresh_classifiers.append(('LightGBM', lgb))
                logging.info("Fresh LightGBM model trained")
            
            if len(fresh_classifiers) < 2:
                raise ValueError("Could not create enough fresh classifiers for voting")
            
            # Create voting classifier
            voting_clf = VotingClassifier(
                estimators=fresh_classifiers,
                voting='hard'
            )
            
            # The individual models are already trained, so we just need to fit the meta-classifier
            voting_clf.fit(X_train, self.split.y_train)
            
            # Log this model too with signature
            self._log_model_with_signature(voting_clf, X_train)
            
            logging.info(f"Fresh voting classifier created with {len(fresh_classifiers)} models")
            return voting_clf
            
        except Exception as e:
            logging.error(f"Error creating fresh voting classifier: {str(e)}")
            raise

    def get_voting_classifier_info(self):
        """
        Get information about the voting classifier setup
        """
        info = {
            'available_models': list(self.models.keys()),
            'has_preprocessed_data': 'X_train' in self.processed_data,
            'has_preprocessors': bool(self.preprocessors),
            'processed_feature_count': len(self.processed_feature_names) if self.processed_feature_names else 0,
            'original_training_data_shape': self.split.X_train.shape,
            'target_classes': len(set(self.split.y_train))
        }
        return info


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def train_votting_classifier(train_out: dict, split: TrainTestSplitData) -> VotingClassifier:
    """
    Train a voting classifier with preprocessed data
    Args:
        train_out: Dictionary containing trained models and preprocessing metadata
        split: TrainTestSplitData containing training data
    Returns:
        A voting classifier
    """
    try:
        logging.info("Starting voting classifier training with preprocessed data...")
        
        # Debug: Print what we received
        logging.info(f"train_out keys: {list(train_out.keys())}")
        
        trainer = TrainVottingClassifier(train_out, split)
        
        # Get info about the setup
        info = trainer.get_voting_classifier_info()
        logging.info(f"Voting classifier setup info: {info}")
        
        voting_clf = trainer.train_voting_classifier()
        
        return voting_clf
        
    except Exception as e:
        logging.error(f"Error in train_voting_classifier step: {str(e)}")
        raise