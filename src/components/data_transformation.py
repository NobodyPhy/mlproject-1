import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_object_filepath: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for giving the data transformation object
        '''
        try:
            numerical_columns = ['writing score', 'reading score']
            categorical_columns = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )


            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipeline, numerical_columns),
                    ('categorical_pipeline', categorical_pipeline, categorical_columns)
                ]
            )

            logging.info("Column Transformer (preprocessor) created")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        '''
        Performs the data transformation
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of train and test data completed")
            logging.info("Obtaining preprocessor object")

            preprocessor_object = self.get_data_transformer_object()
            target_column_name = 'math score'

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessor object on training and testing dataframe")

            input_features_train_array = preprocessor_object.fit_transform(input_feature_train_df)
            input_features_test_array = preprocessor_object.transform(input_feature_test_df)

            train_array = np.c_[input_features_train_array, np.array(target_feature_train_df)]
            test_array = np.c_[input_features_test_array, np.array(target_feature_test_df)]

            
            logging.info("Data transformation DONE")
            logging.info("Saving preprocessor object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_object_filepath,
                object_ = preprocessor_object
            )

            return(
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_object_filepath
            )
            
        except Exception as e:
            raise CustomException(e, sys)