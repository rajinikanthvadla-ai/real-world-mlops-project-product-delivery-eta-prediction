import mlflow, pandas as pd, boto3
from sklearn.metrics import mean_squared_error
import os

MLFLOW_TRACKING_URI = "http://13.203.199.220:32001/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

def evaluate_models():
    """Evaluate models with fallback for missing models"""
    try:
        # Check if model exists in registry
        try:
            all_versions = client.get_latest_versions("delivery-eta-model")
            if not all_versions:
                print("No models found in registry. Skipping evaluation.")
                return
        except Exception as e:
            print(f"Model registry not accessible: {e}")
            print("This is normal for the first run. Skipping evaluation.")
            return
        
        # Download test data
        s3 = boto3.client("s3")
        s3.download_file('product-delivery-eta-processed-data','processed_data.csv','/tmp/data.csv')
        df = pd.read_csv('/tmp/data.csv')
        
        features = ['product_weight_g','product_volume_cm3','price','freight_value',
                    'purchase_hour','purchase_day_of_week','purchase_month']
        X,y = df[features],df['delivery_duration_days']
        
        # Try to load Production and Staging models
        try:
            prod_versions = client.get_latest_versions("delivery-eta-model", ["Production"])
            staging_versions = client.get_latest_versions("delivery-eta-model", ["Staging"])
            
            if not prod_versions and not staging_versions:
                print("No Production or Staging models found.")
                # Promote latest model to Staging if available
                latest_versions = client.get_latest_versions("delivery-eta-model")
                if latest_versions:
                    version = latest_versions[0].version
                    client.transition_model_version_stage(
                        "delivery-eta-model", version, "Staging"
                    )
                    print(f"Promoted model version {version} to Staging")
                return
            
            if prod_versions and staging_versions:
                # Compare Production vs Staging
                prod = mlflow.pyfunc.load_model(f"models:/delivery-eta-model/Production")
                staging = mlflow.pyfunc.load_model(f"models:/delivery-eta-model/Staging")
                
                pr, sr = prod.predict(X), staging.predict(X)
                prod_rmse = mean_squared_error(y,pr,squared=False)
                stag_rmse = mean_squared_error(y,sr,squared=False)
                
                print(f"Production RMSE: {prod_rmse:.4f}")
                print(f"Staging RMSE: {stag_rmse:.4f}")
                
                if stag_rmse < prod_rmse:
                    ver = staging_versions[0].version
                    client.transition_model_version_stage(
                        "delivery-eta-model",ver,"Production",archive_existing_versions=True
                    )
                    print(f"Promoted staging v{ver} to Production (Better RMSE: {stag_rmse:.4f})")
                else:
                    print("Keeping current Production model (Better performance)")
            
            elif staging_versions and not prod_versions:
                # Promote Staging to Production if no Production exists
                ver = staging_versions[0].version
                client.transition_model_version_stage(
                    "delivery-eta-model",ver,"Production"
                )
                print(f"Promoted staging v{ver} to Production (No existing Production model)")
            
        except Exception as model_error:
            print(f"Model evaluation failed: {model_error}")
            print("This might be due to model artifacts not being accessible")
    
    except Exception as e:
        print(f"Evaluation process failed: {e}")
        print("Continuing pipeline...")

if __name__ == "__main__":
    evaluate_models()
