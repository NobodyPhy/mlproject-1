# Basic Import
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import warnings

# Modelling
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trainer_model_filepath:str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self,):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
            "Linear Regression": LinearRegression(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGB Regressor": XGBRegressor(),
            "Cat Boosting Regressor": CatBoostRegressor(verbose=False),
            "Ada Boost Regressor": AdaBoostRegressor(),
            }

            logging.info("Fitting and evaluating all models")
            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model was found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trainer_model_filepath,
                object_ = best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2


        except Exception as e:
            CustomException(e, sys)