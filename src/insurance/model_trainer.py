import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from insurance.config import ProjectConfig

class ModelTrainer:
    """Trains and evaluates a LightGBM regression model based on project config."""

    def __init__(self, df: pd.DataFrame, config: ProjectConfig):
        self.df = df
        self.config = config
        self.model = lgb.LGBMRegressor(force_col_wise=True)
        self.best_model = None

    def split(self):
        X = self.df.drop(columns=[self.config.target])
        y = self.df[self.config.target]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def tune_hyperparameters(self, X_train, y_train):
        param_grid = {
            "learning_rate": self.config.parameters["learning_rate"],
            "n_estimators": self.config.parameters["n_estimators"],
            "num_leaves": self.config.parameters["num_leaves"]
        }

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring="r2",
            cv=5
        )
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_

    def train_final_model(self, X_train, y_train):
        self.best_model.fit(X_train, y_train, feature_name=X_train.columns.tolist())
        return self.best_model

    def evaluate(self, y_true, y_pred):
        return {
            "r2": r2_score(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred)
        }