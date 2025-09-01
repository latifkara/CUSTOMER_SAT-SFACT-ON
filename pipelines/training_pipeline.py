from zenml import pipeline  
from steps.ingestData import ingest_data
from steps.splitCategory import split_category
from steps.HandleCategoricalMissingValues import handle_categorical_missing_values
from steps.HandleNumericalMissingValue import handle_numerical_missing_values
from steps.HandleOutlier import handle_outlier
from steps.RobustScaler import robust_scaler
from steps.MergeCols import merge_cols
from steps.LabelEncoderWrapper import label_encoder_wrapper
from steps.OneHotEncoder import one_hot_encoder
from steps.TrainModel import train_model
from steps.TrainVottingClassifier import train_votting_classifier
from steps.EvaulateModel import evaluate_model


@pipeline()
def training_pipeline(data_path: str):
    ingest_data = ingest_data(data_path)
    split_category = split_category(ingest_data)

    handle_numerical_missing_values = handle_numerical_missing_values(ingest_data[split_category['num_cols']])
    handle_outlier = handle_outlier(handle_numerical_missing_values)
    robust_scaler = robust_scaler(handle_outlier)
    
    handle_categorical_missing_values = handle_categorical_missing_values(ingest_data[split_category['cat_cols']])
    merge_cols = merge_cols(handle_categorical_missing_values, split_category['cat_cols'], split_category['num_cols'], split_category['cat_but_car'])
    
    label_encoder_wrapper = label_encoder_wrapper(merge_cols, merge_cols[split_category["cat_cols"]])
    one_hot_encoder = one_hot_encoder(merge_cols, merge_cols[split_category["cat_cols"]])

    train_model, X_train, X_test, y_train, y_test = train_model(one_hot_encoder)
    train_votting_classifier = train_votting_classifier(train_model, X_train, y_train)
    evaluate_model = evaluate_model(train_votting_classifier, X_test, y_test)
    
    return evaluate_model