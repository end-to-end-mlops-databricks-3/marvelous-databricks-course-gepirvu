import pandas as pd
from pyspark.sql import SparkSession

class InsuranceDataProcessor:
    def __init__(self, spark: SparkSession, config):
        self.spark = spark
        self.config = config
        self.dataset = None

    def load_data(self) -> pd.DataFrame:
        df = self.spark.read.format("csv") \
            .option("inferSchema", "true") \
            .option("header", "true") \
            .option("sep", ",") \
            .load(self.config.dbfs_path)
        
        self.dataset = df.toPandas()
        return self.dataset

    def preprocess(self) -> pd.DataFrame:
        self.dataset['sex'] = self.dataset['sex'].apply(lambda x: 0 if x == 'female' else 1)
        self.dataset['smoker'] = self.dataset['smoker'].apply(lambda x: 0 if x == 'no' else 1)
        region_dummies = pd.get_dummies(self.dataset['region'], drop_first=True)
        self.dataset = pd.concat([self.dataset.drop('region', axis=1), region_dummies], axis=1)
        self.feature_names = self.dataset.drop(columns=["charges"]).columns.tolist()

        return self.dataset