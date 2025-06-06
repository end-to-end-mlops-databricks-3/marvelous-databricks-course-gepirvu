# src/insurance/config.py
"""Configuration module for the insurance project.

Defines the structure and loading mechanism for project configuration
such as features, target, catalog names, and experiment parameters.
"""

from typing import Any

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    """Represent project configuration parameters loaded from YAML.

    Handles feature specifications, catalog details, and experiment parameters.
    Supports environment-specific configuration overrides.
    """

    num_features: list[str]
    cat_features: list[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: dict[str, Any]
    data_path: str | None = None
    experiment_name_basic: str | None
    experiment_name_custom: str | None
    experiment_name_fe: str | None

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load configuration from a YAML file for a specific environment.

        Args:
            config_path: Path to the YAML configuration file.
            env: Environment name ('dev', 'acc', or 'prd').

        Returns:
            ProjectConfig: An instance of the project configuration.

        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        env_config = config_dict.get(env, {})
        merged_config = {**{k: v for k, v in config_dict.items() if k not in ["dev", "acc", "prd"]}, **env_config}
        return cls(**merged_config)


class Tags(BaseModel):
    """Represents a set of tags for a Git commit.

    Contains information about the Git SHA, branch, and job run ID.
    """

    git_sha: str
    branch: str
