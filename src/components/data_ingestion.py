import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
 
class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            # Read the dataset
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Dataset read as dataframe")

            # Create the artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.config.raw_data_path, index=False)
            logging.info("Raw data saved")

            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save the train and test sets
            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)

            logging.info("Train and test data saved")
            return (
                self.config.train_data_path,
                self.config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    preprocessor = data_transformation.initiate_data_transformation(train_data, test_data)



