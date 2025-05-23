"""Conftest module."""

from insurance import PROJECT_DIR

# Define directories
MLRUNS_DIR = PROJECT_DIR / "tests" / "mlruns"
CATALOG_DIR = PROJECT_DIR / "tests" / "catalog"

# Ensure directories exist
CATALOG_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

# Use URI-safe absolute path
TRACKING_URI = MLRUNS_DIR.resolve().as_uri()

# Register fixtures
pytest_plugins = [
    "tests.fixtures.datapreprocessing_fixture",
    "tests.fixtures.custom_model_fixture",
]

