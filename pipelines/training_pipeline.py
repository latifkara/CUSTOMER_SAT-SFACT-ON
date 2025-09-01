# training_pipeline.py
from zenml import pipeline  
from steps.ingestData import ingest_data
from steps.splitCategory import split_category, get_numerical_data, get_categorical_data
from steps.HandleCategoricalMissingValues import handle_categorical_missing_values
from steps.HandleNumericalMissingValue import handle_numerical_missing_values
from steps.HandleOutlier import handle_outlier
from steps.RobustScaler import robust_scaler
from steps.MergeCols import merge_cols
from steps.LabelEncoderWrapper import label_encoder_wrapper
from steps.OneHotEncoder import one_hot_encoder
from steps.SplitData import split_data
from steps.TrainModel import train_model
from steps.TrainVottingClassifier import train_votting_classifier
from steps.EvaulateModel import evaluate_model


@pipeline
def training_pipeline(data_path: str):
    # Ingest data
    ingest_data_out = ingest_data(data_path)
    
    # Split columns into different categories (returns a dictionary)
    column_info = split_category(ingest_data_out)
    
    # Extract numerical and categorical data
    numerical_data = get_numerical_data(ingest_data_out, column_info)
    categorical_data = get_categorical_data(ingest_data_out, column_info)
    
    # Process numerical data
    handle_numerical_missing_value_out = handle_numerical_missing_values(numerical_data)
    handle_outlier_out = handle_outlier(handle_numerical_missing_value_out)
    robust_scaler_out = robust_scaler(handle_outlier_out)
    
    # Process categorical data
    handle_categorical_missing_value_out = handle_categorical_missing_values(categorical_data)
    
    # Merge processed data
    merge_cols_out = merge_cols(robust_scaler_out, handle_categorical_missing_value_out)
    
    # Encode categorical variables - you'll need to pass the column info to these steps too
    label_encoder_wrapper_out = label_encoder_wrapper(merge_cols_out, column_info)
    one_hot_encoder_out = one_hot_encoder(label_encoder_wrapper_out, column_info)

    # Split data into X_train, X_test, y_train, y_test
    train_test_split = split_data(one_hot_encoder_out, "TARGET")

    # Train models
    train_model_out = train_model(train_test_split)
    train_votting_classifier_out = train_votting_classifier(train_model_out, train_test_split)
    
    # Evaluate model
    evaluate_model_out = evaluate_model(train_votting_classifier_out, train_test_split)

    return evaluate_model_out