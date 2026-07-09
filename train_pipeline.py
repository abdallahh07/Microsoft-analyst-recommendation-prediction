import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from microsoft_model.config.config import config
from microsoft_model.processing.features import merge_all
from microsoft_model.pipeline import microsoft_pipeline

def run_training():
    # Load data
    recommendations, income, balance, cashflow, price = load_data()
    
    # Build features and merge
    df = merge_all(recommendations, income, balance, cashflow, price)
    
    # Split
    X = df[config.numerical_features]
    y = df[config.target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )
    
    # Train
    microsoft_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = microsoft_pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC: {auc:.4f}")
    
    # Save
    save_path = Path(__file__).parent / config.pipeline_save_file
    joblib.dump(microsoft_pipeline, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    run_training()