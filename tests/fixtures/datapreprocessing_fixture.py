"""Dataloader fixture."""

import pandas as pd
import pytest
from loguru import logger
from pyspark.sql import SparkSession

from insurance import PROJECT_DIR
from insurance.config import ProjectConfig, Tags
from tests.unit_tests.spark_config import spark_config
from collections.abc import Generator
from databricks.connect import DatabricksSession 


@pytest.fixture(scope="session")
def spark_session() -> Generator[DatabricksSession, None, None]:
    """Creates a Spark Connect session to Databricks."""
    spark = (
        DatabricksSession.builder
        .remote(spark_config.host)
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def config() -> ProjectConfig:
    """Load and return the project configuration.

    This fixture reads the project configuration from a YAML file and returns a ProjectConfig object.

    :return: The loaded project configuration
    """
    config_file_path = (PROJECT_DIR / "project_config.yml").resolve()
    logger.info(f"Current config file path: {config_file_path.as_posix()}")
    config = ProjectConfig.from_yaml(config_file_path.as_posix())
    return config


@pytest.fixture(scope="function")
def sample_data(config: ProjectConfig, spark_session: SparkSession) -> pd.DataFrame:
    """Create a sample DataFrame from a CSV file.

    This fixture reads a CSV file using either Spark or pandas, then converts it to a Pandas DataFrame,

    :return: A sampled Pandas DataFrame containing some sample of the original data.
    """
    file_path = PROJECT_DIR / "tests" / "test_data" / "sample.csv"
    sample = pd.read_csv(file_path.as_posix(), sep=",")

    # Alternative approach to reading the sample
    # Important Note: Replace NaN with None in Pandas Before Conversion to Spark DataFrame:
    # sample = sample.where(sample.notna(), None)  # noqa
    # sample = spark_session.createDataFrame(sample).toPandas()  # noqa
    return sample


@pytest.fixture(scope="session")
def tags() -> Tags:
    """Create and return a Tags instance for the test session.

    This fixture provides a Tags object with predefined values for git_sha, branch, and job_run_id.
    """
    return Tags(git_sha="wxyz", branch="test", job_run_id="9")