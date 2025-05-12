# Databricks notebook source
# %pip install -e ../src
# %restart_python

# COMMAND ----------
# Manually add src path if not using pip install
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent / "src"))

# COMMAND ----------
from loguru import logger
from pyspark.sql import SparkSession
import pandas as pd

from insurance.config import InsuranceConfig
from insurance.data_preprocessing import InsuranceDataProcessor
from insurance.model_trainer import InsuranceModelTrainer

# Optional logging/timer utilities
from marvelous.logging import setup_logging
from marvelous.timer import Timer

# Setup logging
setup_logging(log_file="logs/insurance_pipeline.log")
logger.info("Pipeline started.")

# Load configuration
config = InsuranceConfig()
logger.info(f"Configuration: {config}")

# COMMAND ----------
# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

# Load & preprocess data
with Timer() as data_timer:
    processor = InsuranceDataProcessor(spark, config)
    df = processor.load_data()
    df = processor.preprocess()
    df.head()

logger.info(f"Data loaded and preprocessed in {data_timer}")

df.head()

# COMMAND ----------
# Train + Evaluate model
trainer = InsuranceModelTrainer(df, config, processor.feature_names)
X_train, X_test, y_train, y_test = trainer.split()

with Timer() as train_timer:
    best_params, best_score = trainer.tune_hyperparameters(X_train, y_train)
    final_model = trainer.train_final_model(X_train, y_train)
    y_pred = final_model.predict(X_test)
    metrics = trainer.evaluate(y_test, y_pred)

logger.info(f"Model trained in {train_timer}")
logger.info(f"Best Params: {best_params}")
logger.info(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}")

# COMMAND ----------
# (Optional) Plot feature importances
import matplotlib.pyplot as plt
import lightgbm as lgb

lgb.plot_importance(final_model)
plt.title("Feature Importance")
plt.show()

# COMMAND ----------
# (Optional) Save model to DBFS or MLflow
# from joblib import dump
# dump(final_model, "/dbfs/FileStore/models/best_lightgbm_model.pkl")

# COMMAND ----------
logger.info("Pipeline finished successfully.")
