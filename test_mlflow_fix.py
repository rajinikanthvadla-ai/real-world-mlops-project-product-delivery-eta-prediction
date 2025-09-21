#!/usr/bin/env python3
"""
Quick test to verify MLflow model registration fix
"""
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature

# Setup MLflow
MLFLOW_TRACKING_URI = "http://13.203.199.220:32001/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("test-model-registration")

print("🧪 Testing MLflow model registration fix...")

# Create dummy data
np.random.seed(42)
X = pd.DataFrame(np.random.rand(100, 7), columns=[
    'product_weight_g','product_volume_cm3','price','freight_value',
    'purchase_hour','purchase_day_of_week','purchase_month'
])
y = np.random.rand(100) * 10  # Random delivery times

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = xgb.XGBRegressor(n_estimators=10, max_depth=3)
model.fit(X_train, y_train)

# Test prediction and signature
predictions = model.predict(X_test)
signature = infer_signature(X_train, predictions)

print("✅ Model trained successfully")

# Test MLflow logging
try:
    with mlflow.start_run() as run:
        mlflow.log_param("test_param", "test_value")
        mlflow.log_metric("test_rmse", 1.23)
        
        # Test the fixed log_model call
        mlflow.xgboost.log_model(
            model, 
            "model",
            signature=signature,
            registered_model_name="test-delivery-eta-model"
        )
        
        print(f"✅ Model logged successfully: {run.info.run_id}")
        
        # Test model registry
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions("test-delivery-eta-model")
        
        if latest_versions:
            version = latest_versions[0].version
            print(f"✅ Model registered successfully: version {version}")
            
            # Test stage transition
            client.transition_model_version_stage(
                "test-delivery-eta-model",
                version,
                "Staging"
            )
            print(f"✅ Model promoted to Staging: version {version}")
        else:
            print("⚠️ Model registration might have failed")

except Exception as e:
    print(f"❌ MLflow test failed: {e}")
    import traceback
    traceback.print_exc()

print("🏁 MLflow test completed")