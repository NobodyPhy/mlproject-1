import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts', "raw.csv")
    train_data_path: str=os.path.join('artifacts', "train.csv")
    test_data_path: str=os.path.join('artifacts', "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        '''
        This function reads and write raw data, performs train-test split and save them to csv 
        '''

        logging.info("Entering the data ingestion method")

        try:
            logging.info("Reading the dataset as dataframe and saving to csv")
            df = pd.read_csv('notebooks\data\StudentsPerformance.csv')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Splitting data to train test")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            logging.info("Saving train and test data to csv")
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e: 
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)