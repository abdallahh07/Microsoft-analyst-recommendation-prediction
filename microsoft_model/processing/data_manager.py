import pandas as pd 
import numpy as np 
from pathlib import Path
from microsoft_model.config.config import config

def load_data() -> pd.DataFrame:
  data_path = Path(__file__).parent.parent.parent/config.data_folder
  
  recommendations = pd.read_csv(data_path/config.recommendations_file)
  income = pd.read_csv(data_path/config.income_file)
  balance = pd.read_csv(data_path/config.balance_file)
  cashflow = pd.read_csv(data_path / config.cashflow_file)
  price = pd.read_csv(data_path / config.price_file)
    
  return recommendations, income, balance, cashflow, price