"""Spark Configuration module for local testing."""

from pydantic_settings import BaseSettings


# Load configuration from environment variables
class SparkConfig(BaseSettings):
    """Configuration for Spark Connect to Databricks."""

    host: str = "sc://dbc-c2e8445d-159d.cloud.databricks.com"
    app_name: str = "insurance_test"


spark_config = SparkConfig()
