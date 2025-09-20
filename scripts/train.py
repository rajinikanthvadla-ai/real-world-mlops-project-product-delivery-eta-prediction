import pandas as pd, boto3, os, mlflow, mlflow.xgboost, xgboost as xgb, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import sys

# Environment detection
IS_SAGEMAKER = '/opt/ml' in os.getcwd() or 'SM_MODEL_DIR' in os.environ
IS_GITHUB_ACTIONS = 'GITHUB_ACTIONS' in os.environ

# MLflow configuration
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', "http://13.127.63.212:32001/")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

if IS_SAGEMAKER:
    mlflow.set_experiment("sagemaker-delivery-eta-prediction")
elif IS_GITHUB_ACTIONS:
    mlflow.set_experiment("github-actions-delivery-eta-prediction")
else:
    mlflow.set_experiment("delivery-eta-prediction")

print(f"🔧 Environment: {'SageMaker' if IS_SAGEMAKER else 'GitHub Actions' if IS_GITHUB_ACTIONS else 'Local'}")
print(f"🔧 MLflow URI: {MLFLOW_TRACKING_URI}")

s3 = boto3.client("s3")
bucket_processed = "product-delivery-eta-processed-data"
key_processed = "processed_data.csv"

local_path = "/tmp/processed_data.csv"

# Step 1: Try downloading processed data
try:
    s3.download_file(bucket_processed, key_processed, local_path)
    print("✅ Loaded processed data from S3")
    df = pd.read_csv(local_path)

except Exception as e:
    print("⚠️ Processed data not found in S3. Building from raw dataset...")
    # Download raw data from raw bucket
    raw_bucket = "product-delivery-eta-raw-data"
    local_dir = "/tmp/raw"
    os.makedirs(local_dir, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=raw_bucket):
        for obj in page.get("Contents", []):
            s3.download_file(raw_bucket, obj["Key"], os.path.join(local_dir, os.path.basename(obj["Key"])))

    # Load datasets
    orders_df = pd.read_csv(f"{local_dir}/olist_orders_dataset.csv")
    order_items_df = pd.read_csv(f"{local_dir}/olist_order_items_dataset.csv")
    products_df = pd.read_csv(f"{local_dir}/olist_products_dataset.csv")
    customers_df = pd.read_csv(f"{local_dir}/olist_customers_dataset.csv")
    sellers_df = pd.read_csv(f"{local_dir}/olist_sellers_dataset.csv")

    # Merge + process
    merged_df = order_items_df.merge(orders_df,on="order_id") \
                              .merge(products_df,on="product_id") \
                              .merge(customers_df,on="customer_id") \
                              .merge(sellers_df,on="seller_id")

    merged_df["order_purchase_timestamp"] = pd.to_datetime(merged_df["order_purchase_timestamp"], errors="coerce")
    merged_df["order_delivered_customer_date"] = pd.to_datetime(merged_df["order_delivered_customer_date"], errors="coerce")
    merged_df = merged_df.dropna(subset=["order_purchase_timestamp","order_delivered_customer_date"])

    merged_df["delivery_duration_days"] = (
        merged_df["order_delivered_customer_date"] - merged_df["order_purchase_timestamp"]
    ).dt.total_seconds() / (24*3600)

    merged_df = merged_df[(merged_df["delivery_duration_days"] > 0) & (merged_df["delivery_duration_days"] < 60)]
    merged_df["purchase_hour"] = merged_df["order_purchase_timestamp"].dt.hour
    merged_df["purchase_day_of_week"] = merged_df["order_purchase_timestamp"].dt.dayofweek
    merged_df["purchase_month"] = merged_df["order_purchase_timestamp"].dt.month
    merged_df["product_volume_cm3"] = merged_df["product_length_cm"] * merged_df["product_height_cm"] * merged_df["product_width_cm"]

    # Save + upload
    merged_df.to_csv(local_path,index=False)
    s3.upload_file(local_path, bucket_processed, key_processed)
    print("✅ Processed data built & uploaded to S3")
    df = merged_df

def train_model(df):
    """Training function that works both locally and in SageMaker"""
    features = ['product_weight_g','product_volume_cm3','price','freight_value',
                'purchase_hour','purchase_day_of_week','purchase_month']
    target = 'delivery_duration_days'
    X,y = df[features], df[target]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    # Parse hyperparameters (for SageMaker)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    args, _ = parser.parse_known_args()
    
    params={"objective":"reg:squarederror",
            "n_estimators":args.n_estimators,
            "max_depth":args.max_depth,
            "learning_rate":args.learning_rate}
    
    model=xgb.XGBRegressor(**params).fit(X_train,y_train)
    preds=model.predict(X_test)
    rmse=mean_squared_error(y_test,preds,squared=False)
    mae=mean_absolute_error(y_test,preds)
    
    if IS_SAGEMAKER:
        # SageMaker: Save model locally, MLflow will track
        model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
        joblib.dump(model, os.path.join(model_dir, 'model.joblib'))
        
        # Log to MLflow (with SageMaker integration)
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics({"rmse":rmse,"mae":mae})
            mlflow.sklearn.log_model(model, "model", registered_model_name="delivery-eta-model")
            
            # Promote to Staging for evaluation
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions("delivery-eta-model", stages=["None"])[0]
            client.transition_model_version_stage(
                name="delivery-eta-model",
                version=model_version.version,
                stage="Staging"
            )
    elif IS_GITHUB_ACTIONS:
        # GitHub Actions: Log directly to MLflow server without S3 artifacts
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics({"rmse":rmse,"mae":mae})
            
            # Create a simple model signature
            from mlflow.models.signature import infer_signature
            signature = infer_signature(X_train, model.predict(X_train))
            
            # Log model without storing large artifacts to S3
            mlflow.xgboost.log_model(
                model, 
                "model",
                signature=signature,
                registered_model_name="delivery-eta-model",
                # Use local artifact store to avoid S3 permissions
                artifact_path="model"
            )
            
            # Promote to Staging automatically for GitHub Actions
            try:
                client = mlflow.tracking.MlflowClient()
                # Get the latest version we just created
                latest_versions = client.get_latest_versions("delivery-eta-model", stages=["None"])
                if latest_versions:
                    model_version = latest_versions[0]
                    client.transition_model_version_stage(
                        name="delivery-eta-model",
                        version=model_version.version,
                        stage="Staging"
                    )
                    print(f"✅ Model v{model_version.version} promoted to Staging")
            except Exception as e:
                print(f"⚠️ Model promotion failed: {e}")
    else:
        # Local training: Use file-based MLflow logging to avoid S3 issues
        local_mlflow_dir = "./mlruns"
        os.makedirs(local_mlflow_dir, exist_ok=True)
        
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics({"rmse":rmse,"mae":mae})
            
            # Save model locally instead of trying to upload to S3
            model_path = f"./models/delivery-eta-model-{run.info.run_id}"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            mlflow.xgboost.save_model(model, model_path)
            
            # Register model manually
            try:
                mlflow.register_model(f"file://{os.path.abspath(model_path)}", "delivery-eta-model")
                print(f"✅ Model registered: delivery-eta-model")
            except Exception as e:
                print(f"⚠️ Model registration failed: {e}")
    
    print(f"✅ Training complete: RMSE={rmse:.4f}, MAE={mae:.4f}")
    return model, rmse, mae

if __name__ == "__main__":
    # Run training
    model, rmse, mae = train_model(df)
