from pipelines.training_pipeline import training_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path=r"F:\KTU_Yuksek_Lisans\1_sınıf_1_dönem\Knowledge_Discovery_in_Large_Data_Sets\Customer_satisfaction_repo\CUSTOMER_SAT-SFACT-ON\data\application_train_sample.csv")



