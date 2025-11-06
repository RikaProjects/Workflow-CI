"""
Modelling for Heart Disease Prediction with MLflow
Nama: Rika Rostika Afipah
Kriteria 2 - Basic Requirements
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Set MLflow tracking
mlflow.set_experiment("Heart_Disease_Prediction_Rika")

def load_data():
    """Load processed data for modelling"""
    print("Loading processed data...")
    data = pd.read_csv('heart_disease_processed.csv')
    print(f"Data loaded: {data.shape}")
    
    X = data.drop('heart_disease', axis=1)
    y = data['heart_disease']
    
    print(f"Features: {X.shape[1]}, Target distribution: {y.value_counts().to_dict()}")
    return X, y

def train_basic_model():
    """Train model with MLflow autolog only (Basic Criteria)"""
    print("Starting model training with MLflow autolog...")
    
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
    
    # Start MLflow run
    with mlflow.start_run(run_name="RandomForest_Basic_Autolog"):
        print("Enabling MLflow autolog...")
        
        # HANYA AUTOLOG
        mlflow.sklearn.autolog()
        
        # Train Random Forest model
        print("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        model.fit(X_train, y_train)
        
        # Predictions untuk print metrics saja
        y_pred = model.predict(X_test)
        
        # Calculate metrics hanya untuk display
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("Model Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        print("Model training completed with MLflow autolog!")
        print("Check MLflow UI: mlflow ui --backend-store-uri file:///mlruns")
        
    return model

if __name__ == "__main__":
    model = train_basic_model()
    
    # Optional: Save model locally (bukan requirement MLflow)
    import joblib
    joblib.dump(model, 'heart_disease_model.pkl')

    print("Model saved locally as 'heart_disease_model.pkl'")
