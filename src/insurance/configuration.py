from dataclasses import dataclass


@dataclass
class InsuranceConfig:
    dbfs_path: str = "dbfs:/Volumes/mlops_dev/pirvugeo/data/insurance.csv"
    test_size: float = 0.2
    random_state: int = 0
    param_grid: dict = None

    def __post_init__(self):
        self.param_grid = {
            "num_leaves": [29, 30, 31],
            "learning_rate": [0.08, 0.1, 0.12],
            "n_estimators": [80, 100, 120],
        }
