import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    model_report_file_path: str = os.path.join('artifacts', 'model_report.txt')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'Decision Tree': DecisionTreeRegressor(),
            'KNN': KNeighborsRegressor(),
            'XGBoost': XGBRegressor(),
            'CatBoost': CatBoostRegressor(verbose=0),
            'Gradient Boosting': GradientBoostingRegressor(),
            'AdaBoost': AdaBoostRegressor()
        }
        self.parameters = {
            'Linear Regression': {
                'fit_intercept': [True],
            },
            'Random Forest': {
                'n_estimators': [10, 50, 100],
                'max_depth': [None, 10, 20, 30]
            },
            'Decision Tree': {
                'max_depth': [None, 10, 20, 30]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1]
            },
            'CatBoost': {
                'iterations': [100, 200],
                'depth': [6, 8]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1]
            },
            'AdaBoost': {
                'n_estimators': [50, 100],
                'learning_rate': [1.0, 1.5]
            }
    
        }
        self.best_model = None

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting Input data into features and target variable")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            model_report = {}

            for model_name, model in self.models.items():

                gs = GridSearchCV(model, self.parameters[model_name], cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)

                # Use the best estimator from GridSearchCV
                model = gs.best_estimator_
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2_square = r2_score(y_test, y_pred)
                model_report[model_name] = r2_square

            # Find the best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            self.best_model = self.models[best_model_name]

            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_model_score}")

            # Save the best model
            save_object(self.model_trainer_config.trained_model_file_path, 
                        self.best_model)

            # Save the model report
            with open(self.model_trainer_config.model_report_file_path, 'w') as f:
                for model_name, score in model_report.items():
                    f.write(f"{model_name}: {score}\n")

        except Exception as e:
            raise CustomException(e, sys)

