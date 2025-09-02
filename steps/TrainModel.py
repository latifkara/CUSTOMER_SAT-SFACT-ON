from tkinter import N
import pandas as pd
from zenml import step
import logging
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from src.TrainTestSplitData import TrainTestSplitData


class TrainModel:
    """
    Train multiple models with categorical data preprocessing and feature selection
    Args:
        split: TrainTestSplitData containing train/test data
    Returns:
        Dictionary of trained models
    """
    def __init__(self, split: TrainTestSplitData):
        self.split = split
        self.models: Dict[str, Any] = {}
        self.importance_features: Dict[str, List[str]] = {}
        self.preprocessors: Dict[str, Any] = {}
        self.processed_feature_names: List[str] = []
        
        # Initialize classifiers
        self.classifiers = [
            ('DT', DecisionTreeClassifier(random_state=42)),
            ('RF', RandomForestClassifier(random_state=42, n_estimators=100)),
            ('XGBoost', XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss', 
                random_state=42,
                verbosity=0,
                enable_categorical=True  # Enable categorical support
            )),
            ('LightGBM', LGBMClassifier(
                random_state=42, 
                verbose=-1,
                objective='binary',  # Adjust based on your problem type
                force_col_wise=True
            ))
        ]

    def identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify numerical and categorical columns
        """
        numerical_cols = []
        categorical_cols = []
        
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                categorical_cols.append(col)
            elif df[col].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']:
                numerical_cols.append(col)
            else:
                # Handle other dtypes
                if df[col].nunique() < 20:  # Assume categorical if few unique values
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
        
        logging.info(f"Identified {len(numerical_cols)} numerical columns and {len(categorical_cols)} categorical columns")
        logging.info(f"Categorical columns: {categorical_cols[:10]}...")  # Show first 10
        
        return numerical_cols, categorical_cols

    def preprocess_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Preprocess the data to handle categorical variables
        """
        try:
            logging.info("Starting data preprocessing...")
            
            # Identify column types
            numerical_cols, categorical_cols = self.identify_column_types(X_train)
            
            if not categorical_cols:
                # No categorical columns, return as is
                logging.info("No categorical columns found, returning original data")
                return X_train.values, X_test.values, list(X_train.columns)
            
            # Handle missing values in categorical columns
            for col in categorical_cols:
                X_train[col] = X_train[col].fillna('Unknown')
                X_test[col] = X_test[col].fillna('Unknown')
            
            # Handle missing values in numerical columns
            for col in numerical_cols:
                X_train[col] = X_train[col].fillna(X_train[col].median())
                X_test[col] = X_test[col].fillna(X_train[col].median())  # Use train median for test
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
                ],
                remainder='passthrough'
            )
            
            # Fit preprocessor on training data
            logging.info("Fitting preprocessor...")
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # Get feature names after preprocessing
            feature_names = []
            
            # Numerical feature names (same as original)
            feature_names.extend(numerical_cols)
            
            # Categorical feature names (after one-hot encoding)
            if categorical_cols:
                cat_encoder = preprocessor.named_transformers_['cat']
                cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
                feature_names.extend(cat_feature_names)
            
            # Store preprocessor for later use
            self.preprocessors['main'] = preprocessor
            
            logging.info(f"Preprocessing completed. Shape: {X_train_processed.shape}")
            logging.info(f"Feature count: {len(feature_names)}")
            
            return X_train_processed, X_test_processed, feature_names
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            # Fallback: try simple label encoding
            return self.simple_preprocessing(X_train, X_test)

    def simple_preprocessing(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Simple preprocessing using label encoding as fallback
        """
        try:
            logging.info("Using simple label encoding as fallback...")
            
            X_train_copy = X_train.copy()
            X_test_copy = X_test.copy()
            
            # Identify categorical columns
            categorical_cols = X_train_copy.select_dtypes(include=['object', 'category']).columns
            
            # Apply label encoding to categorical columns
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                
                # Handle missing values
                X_train_copy[col] = X_train_copy[col].fillna('Unknown')
                X_test_copy[col] = X_test_copy[col].fillna('Unknown')
                
                # Fit on combined data to ensure consistent encoding
                combined_data = pd.concat([X_train_copy[col], X_test_copy[col]])
                le.fit(combined_data)
                
                X_train_copy[col] = le.transform(X_train_copy[col])
                X_test_copy[col] = le.transform(X_test_copy[col])
                
                label_encoders[col] = le
            
            # Store encoders
            self.preprocessors['label_encoders'] = label_encoders
            
            # Handle missing values in numerical columns
            numerical_cols = X_train_copy.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                X_train_copy[col] = X_train_copy[col].fillna(X_train_copy[col].median())
                X_test_copy[col] = X_test_copy[col].fillna(X_train_copy[col].median())
            
            logging.info("Simple preprocessing completed")
            return X_train_copy.values, X_test_copy.values, list(X_train_copy.columns)
            
        except Exception as e:
            logging.error(f"Error in simple preprocessing: {str(e)}")
            raise

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
            feature_imp = pd.DataFrame({
                'Value': importance, 
                'Feature': feature_names
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
            return feature_names[:num]

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
                verbosity=0,
                enable_categorical=True
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
        Train all models with preprocessing and feature selection
        Returns:
            Dictionary of trained models with metadata
        """
        try:
            logging.info("Starting model training process...")
            logging.info(f"Original training data shape: {self.split.X_train.shape}")
            logging.info(f"Original test data shape: {self.split.X_test.shape}")
            
            # Preprocess the data
            X_train_processed, X_test_processed, feature_names = self.preprocess_data(
                self.split.X_train.copy(), 
                self.split.X_test.copy()
            )
            
            # Store processed feature names
            self.processed_feature_names = feature_names
            
            logging.info(f"Processed training data shape: {X_train_processed.shape}")
            logging.info(f"Processed test data shape: {X_test_processed.shape}")
            
            # Step 1: Train initial models to get feature importance
            logging.info("Step 1: Training models for feature importance...")
            initial_models = {}
            
            for name, classifier in self.classifiers:
                logging.info(f"Training initial {name} model...")
                try:
                    model = classifier.fit(X_train_processed, self.split.y_train)
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
                        selected_indices = [feature_names.index(fname) for fname in selected_feature_names if fname in feature_names]
                        
                        if not selected_indices:
                            logging.warning(f"No valid feature indices found for {name}, using all features")
                            X_train_selected = X_train_processed
                        else:
                            X_train_selected = X_train_processed[:, selected_indices]
                        
                        # Create new instance of classifier for retraining
                        new_classifier = self.create_fresh_classifier(name)
                        
                        # Train with selected features
                        trained_model = new_classifier.fit(X_train_selected, self.split.y_train)
                        
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
                'preprocessors': self.preprocessors,
                'processed_feature_names': self.processed_feature_names,
                'processed_data': {
                    'X_train': X_train_processed,
                    'X_test': X_test_processed
                }
            }
            
        except Exception as e:
            logging.error(f"Error in train_model: {str(e)}")
            raise

    def get_model_performance(self, X_train_processed: np.ndarray) -> Dict[str, float]:
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
                selected_indices = [self.processed_feature_names.index(fname) 
                                  for fname in selected_feature_names 
                                  if fname in self.processed_feature_names]
                
                if selected_indices:
                    X_selected = X_train_processed[:, selected_indices]
                else:
                    X_selected = X_train_processed
                
                y_pred = model.predict(X_selected)
                accuracy = accuracy_score(self.split.y_train, y_pred)
                performance[name] = accuracy
                
                logging.info(f"{name} training accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logging.error(f"Error evaluating {name}: {str(e)}")
                performance[name] = 0.0
        
        return performance


@step
def train_model(split: TrainTestSplitData) -> Dict[str, Any]:
    """
    Train multiple models with preprocessing and feature selection
    Args:
        split: TrainTestSplitData containing X_train, X_test, y_train, y_test
    Returns:
        Dictionary containing trained models, preprocessing info, and performance metrics
    """
    try:
        logging.info("Initializing TrainModel with preprocessing...")
        trainer = TrainModel(split)
        
        # Train the models
        result = trainer.train_model()
        
        # Log performance
        performance = trainer.get_model_performance(result['processed_data']['X_train'])
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