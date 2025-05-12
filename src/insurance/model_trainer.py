import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class InsuranceModelTrainer:
    def __init__(self, dataset: pd.DataFrame, config, feature_names: list[str]):
        self.config = config
        self.feature_names = feature_names
        self.dataset = dataset
        self.model = lgb.LGBMRegressor(force_col_wise=True)
        self.best_model = None

    def split(self):
        X = self.dataset.drop(columns=["charges"]).values
        y = self.dataset["charges"].values
        return train_test_split(X, y, test_size=self.config.test_size, random_state=self.config.random_state)

    def evaluate(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return {"r2": r2, "rmse": rmse, "mae": mae}

    def run_cross_val(self, X, y):
        scores = cross_val_score(self.model, X, y, cv=10, scoring="r2")
        return {"mean_r2": scores.mean(), "std_r2": scores.std()}

    def tune_hyperparameters(self, X, y):
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.config.param_grid, scoring="r2", cv=10)
        grid_search.fit(X, y)
        self.best_model = lgb.LGBMRegressor(**grid_search.best_params_)
        return grid_search.best_params_, grid_search.best_score_

    def train_final_model(self, X_train, y_train):
        self.best_model.fit(X_train, y_train, feature_name=self.feature_names)
        return self.best_model
