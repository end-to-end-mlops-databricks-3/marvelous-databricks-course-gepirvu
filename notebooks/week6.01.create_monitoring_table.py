# Databricks notebook source
# %pip install --upgrade --force-reinstall /dbfs/tmp/insurance-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
import numpy as np
from pyspark.sql import SparkSession
from loguru import logger
from insurance.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
num_features = config.num_features
cat_features = config.cat_features

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()
# COMMAND ----------
from insurance.data_preprocessing import generate_synthetic_data_insurance

inference_data_skewed = generate_synthetic_data_insurance(train_set, drift= True, num_rows=200)
inference_data_skewed = inference_data_skewed.drop(columns=["update_timestamp_utc"])
inference_data_skewed.head(5)

inference_data_skewed_spark = spark.createDataFrame(inference_data_skewed).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

inference_data_skewed_spark.write.mode("overwrite").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.inference_data_skewed"
)


# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Encode categorical and datetime variables
def preprocess_data(df):
    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'datetime']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

train_set, label_encoders = preprocess_data(train_set)

# Define features and target (adjust columns accordingly)
features = train_set.drop(columns=["Id", "charges", "update_timestamp_utc"])
target = train_set["charges"]

# Train a Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(features, target)

# COMMAND ----------
# Identify the most important features
feature_importances = pd.DataFrame({
    'Feature': features.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Top 5 important features:")
print(feature_importances.head(5))

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient

workspace = WorkspaceClient()

#write into feature table; update online table
import time
from databricks.sdk import WorkspaceClient
workspace = WorkspaceClient()
# COMMAND ----------
spark.sql(f"""
    INSERT INTO {config.catalog_name}.{config.schema_name}.insurance_features
    SELECT Id, age, bmi, children
    FROM {config.catalog_name}.{config.schema_name}.inference_data_skewed
""")

# COMMAND ----------
online_table_name = f"{config.catalog_name}.{config.schema_name}.insurance_features_online"

existing_table = workspace.online_tables.get(online_table_name)
logger.info("Online table already exists. Inititating table update.")
pipeline_id = existing_table.spec.pipeline_id
update_response = workspace.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=False)
#update_response = workspace.pipelines.start_update(
#    pipeline_id=pipeline_id, full_refresh=False)
while True:
    update_info = workspace.pipelines.get_update(pipeline_id=pipeline_id, 
                            update_id=update_response.update_id)
    state = update_info.update.state.value
    if state == 'COMPLETED':
        break
    elif state in ['FAILED', 'CANCELED']:
        raise SystemError("Online table failed to update.")
    elif state == 'WAITING_FOR_RESOURCES':
        print("Pipeline is waiting for resources.")
    else:
        print(f"Pipeline is in {state} state.")
    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Send Data to the Endpoint

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
import numpy as np
import datetime
import itertools
from pyspark.sql import SparkSession

from insurance.config import ProjectConfig

# Load configuration

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set") \
                        .withColumn("Id", col("Id").cast("string")) \
                        .toPandas()


inference_data_skewed = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed") \
                        .withColumn("Id", col("Id").cast("string")) \
                        .toPandas()
# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import requests
import time

workspace = WorkspaceClient()
# COMMAND ----------
# Required columns for inference
required_columns = ["Id", "sex", "smoker", "region", "age", "bmi", "children"]

# Sample records from inference datasets
sampled_skewed_records = inference_data_skewed[required_columns].to_dict(orient="records")
test_set_records = test_set[required_columns].to_dict(orient="records")
# COMMAND ----------
# Two different way to send request to the endpoint
# 1. Using https endpoint
def send_request_https(dataframe_record):
    model_serving_endpoint = f"https://{host}/serving-endpoints/insurance-model-serving-fe/invocations"
    response = requests.post(
        model_serving_endpoint,
        headers={"Authorization": f"Bearer {token}"},
        json={"dataframe_records": [dataframe_record]},
    )
    return response

# 2. Using workspace client
def send_request_workspace(dataframe_record):
    response = workspace.serving_endpoints.query(
        name="insurance-model-serving-fe",
        dataframe_records=[dataframe_record]
    )
    return response

# COMMAND ----------
# Loop over test records and send requests for 10 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=10)
for index, record in enumerate(itertools.cycle(test_set_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for test data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)

# COMMAND ----------

# Loop over skewed records and send requests for 10 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=10)
for index, record in enumerate(itertools.cycle(sampled_skewed_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for skewed data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Refresh Monitoring

# COMMAND ----------

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient

from insurance.config import ProjectConfig
from insurance.monitoring import create_or_refresh_monitoring

spark = DatabricksSession.builder.getOrCreate()
workspace = WorkspaceClient()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)