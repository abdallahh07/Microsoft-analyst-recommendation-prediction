from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from microsoft_model.config.config import config

scale_pos_weight = 14

microsoft_pipeline = Pipeline([
  ("scaler",StandardScaler()),
  ("model", lgb.LGBMClassifier(
      scale_pos_weight=scale_pos_weight,
      random_state=config.random_state,
      verbose=-1))]) 