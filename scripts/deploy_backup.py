"""
Alternative deployment script for when MLflow is not available
Uses locally saved models and deploys to SageMaker
"""
import boto3
import os
import time
import joblib
import tarfile
import json

def create_inference_script():
    """Create inference.py for XGBoost SageMaker deployment"""
    inference_code = '''
import xgboost as xgb
import pandas as pd
import os
import io

def model_fn(model_dir):
    """Load XGBoost model"""
    model_path = os.path.join(model_dir, "model.joblib")
    import joblib
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Parse input data for XGBoost"""
    if request_content_type == "text/csv":
        # Parse CSV input - expecting comma-separated values
        values = [float(x.strip()) for x in request_body.split(',')]
        features = ['product_weight_g','product_volume_cm3','price','freight_value',
                   'purchase_hour','purchase_day_of_week','purchase_month']
        df = pd.DataFrame([values], columns=features)
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    predictions = model.predict(input_data)
    return predictions

def output_fn(prediction, content_type):
    """Format output"""
    return str(prediction[0])
'''
    
    with open("/tmp/inference.py", "w") as f:
        f.write(inference_code)
    print("Created XGBoost inference.py")

def deploy_local_model():
    """Deploy locally saved model to SageMaker"""
    
    # Check for local model
    local_model_path = "./models/latest_model.joblib"
    if not os.path.exists(local_model_path):
        print(f"No local model found at {local_model_path}")
        print("Run training first to create a model")
        return False
    
    print(f"Found local model: {local_model_path}")
    
    # AWS setup
    sm = boto3.client("sagemaker", region_name="ap-south-1")
    account = boto3.client("sts").get_caller_identity()["Account"]
    role = f"arn:aws:iam::{account}:role/service-role/AmazonSageMaker-ExecutionRole"
    bucket = "product-delivery-eta-model-artifacts"
    
    try:
        # Create model package
        print("Creating model package...")
        
        # Create inference script
        create_inference_script()
        
        # Create model tarball for XGBoost
        with tarfile.open("/tmp/model.tar.gz", "w:gz") as tar:
            tar.add(local_model_path, arcname="model.joblib")
            # XGBoost container expects model at root level
        
        # Upload to S3
        model_key = f"backup-models/delivery-eta-{int(time.time())}/model.tar.gz"
        s3_model_path = f"s3://{bucket}/{model_key}"
        
        boto3.client("s3").upload_file("/tmp/model.tar.gz", bucket, model_key)
        print(f"Model uploaded to {s3_model_path}")
        
        # Create SageMaker model
        model_name = f"delivery-eta-backup-{int(time.time())}"
        
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": "683313688378.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-xgboost:1.7-1",
                "ModelDataUrl": s3_model_path,
                "Mode": "SingleModel"
            },
            ExecutionRoleArn=role
        )
        
        # Create endpoint configuration
        config_name = f"{model_name}-config"
        sm.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": "ml.t2.medium",
                    "InitialVariantWeight": 1.0
                }
            ]
        )
        
        # Deploy endpoint
        endpoint_name = "delivery-eta-endpoint"
        existing_endpoints = sm.list_endpoints(NameContains=endpoint_name)["Endpoints"]
        
        if existing_endpoints:
            print(f"Updating endpoint: {endpoint_name}")
            sm.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
        else:
            print(f"Creating endpoint: {endpoint_name}")
            sm.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
        
        print("Waiting for deployment...")
        waiter = sm.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        
        print(f"Backup model deployed to SageMaker endpoint: {endpoint_name}")
        return True
        
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting backup model deployment...")
    deploy_local_model()