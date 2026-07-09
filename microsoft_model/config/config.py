from pathlib import Path
from typing import List
from pydantic import BaseModel
import yaml

PACKAGE_ROOT = Path(__file__).parent.parent.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "microsoft_model" / "config" / "config.yml"

class AppConfig(BaseModel):
    package_name: str
    pipeline_name: str
    pipeline_save_file: str
    data_folder: str
    recommendations_file: str
    price_file: str
    income_file: str
    balance_file: str
    cashflow_file: str
    test_size: float
    random_state: int
    target: str
    wacc: float
    growth_rate: float
    numerical_features: List[str]
    features_to_drop: List[str]

def fetch_config_from_yaml() -> AppConfig:
    with open(CONFIG_FILE_PATH, "r") as f:
        parsed = yaml.safe_load(f)
    return AppConfig(**parsed)

config = fetch_config_from_yaml()