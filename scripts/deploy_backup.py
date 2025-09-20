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
    """Create inference.py for SageMaker deployment"""
    inference_code = '''
import joblib
import json
import os

def model_fn(model_dir):
    """Load model for inference"""
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    import pandas as pd
    
    # Expected features
    features = ['product_weight_g','product_volume_cm3','price','freight_value',
                'purchase_hour','purchase_day_of_week','purchase_month']
    
    # Convert to DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = pd.DataFrame(input_data)
    
    # Make prediction
    predictions = model.predict(df[features])
    return predictions.tolist()

def output_fn(prediction, content_type):
    """Format output"""
    if content_type == "application/json":
        return json.dumps({"predictions": prediction})
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
'''
    
    with open("/tmp/inference.py", "w") as f:
        f.write(inference_code)
    print("✅ Created inference.py")

def deploy_local_model():
    """Deploy locally saved model to SageMaker"""
    
    # Check for local model
    local_model_path = "./models/latest_model.joblib"
    if not os.path.exists(local_model_path):
        print(f"❌ No local model found at {local_model_path}")
        print("💡 Run training first to create a model")
        return False
    
    print(f"📦 Found local model: {local_model_path}")
    
    # AWS setup
    sm = boto3.client("sagemaker", region_name="ap-south-1")
    account = boto3.client("sts").get_caller_identity()["Account"]
    role = f"arn:aws:iam::{account}:role/service-role/AmazonSageMaker-ExecutionRole"
    bucket = "product-delivery-eta-model-artifacts"
    
    try:
        # Create model package
        print("🔧 Creating model package...")
        
        # Create inference script
        create_inference_script()
        
        # Create model tarball
        with tarfile.open("/tmp/model.tar.gz", "w:gz") as tar:
            tar.add(local_model_path, arcname="model.joblib")
            tar.add("/tmp/inference.py", arcname="code/inference.py")
        
        # Upload to S3
        model_key = f"backup-models/delivery-eta-{int(time.time())}/model.tar.gz"
        s3_model_path = f"s3://{bucket}/{model_key}"
        
        boto3.client("s3").upload_file("/tmp/model.tar.gz", bucket, model_key)
        print(f"📤 Model uploaded to {s3_model_path}")
        
        # Create SageMaker model
        model_name = f"delivery-eta-backup-{int(time.time())}"
        
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": "246618743249.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3",  # Public SageMaker image
                "ModelDataUrl": s3_model_path,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code"
                }
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
            print(f"📝 Updating endpoint: {endpoint_name}")
            sm.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
        else:
            print(f"🚀 Creating endpoint: {endpoint_name}")
            sm.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
        
        print("⏳ Waiting for deployment...")
        waiter = sm.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        
        print(f"✅ Backup model deployed to SageMaker endpoint: {endpoint_name}")
        return True
        
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔄 Starting backup model deployment...")
    deploy_local_model()