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
import xgboost as xgb
import sagemaker
from sagemaker.xgboost.model import XGBoostModel

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

def _resolve_sagemaker_role() -> str:
    """Resolve an execution role ARN usable by SageMaker."""
    # 1) Allow explicit override via env var
    env_role = os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN")
    if env_role:
        print(f"Using IAM role from env: {env_role}")
        return env_role
    # 2) Try native helper when running inside SageMaker
    try:
        return sagemaker.get_execution_role()
    except Exception:
        pass
    # 3) Try to find a role that looks like a SageMaker execution role
    try:
        iam = boto3.client("iam")
        marker = None
        candidates = []
        while True:
            kwargs = {"Marker": marker} if marker else {}
            resp = iam.list_roles(**kwargs)
            for r in resp.get("Roles", []):
                name = r.get("RoleName", "")
                if name.startswith("AmazonSageMaker-ExecutionRole"):
                    candidates.append(r["Arn"])
            if resp.get("IsTruncated"):
                marker = resp.get("Marker")
            else:
                break
        if candidates:
            print(f"Using discovered IAM role: {candidates[0]}")
            return candidates[0]
    except Exception:
        pass
    # 4) Fallback to the default naming (may fail if it doesn't exist)
    account = boto3.client("sts").get_caller_identity()["Account"]
    fallback = f"arn:aws:iam::{account}:role/service-role/AmazonSageMaker-ExecutionRole"
    print(f"Using fallback IAM role: {fallback}")
    return fallback


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
    role = _resolve_sagemaker_role()
    # Use SageMaker default bucket to avoid cross-bucket permissions issues
    bucket = sagemaker.Session().default_bucket()
    
    try:
        # Create model package
        print("Creating model package...")
        
        # Create inference script
        create_inference_script()
        
        # Package the trained model as joblib for script-mode inference
        model_tar_source = local_model_path
        model_tar_name = "model.joblib"
        
        # Create model tarball
        with tarfile.open("/tmp/model.tar.gz", "w:gz") as tar:
            tar.add(model_tar_source, arcname=model_tar_name)
        
        # Upload to S3
        model_key = f"backup-models/delivery-eta-{int(time.time())}/model.tar.gz"
        s3_model_path = f"s3://{bucket}/{model_key}"
        
        boto3.client("s3").upload_file("/tmp/model.tar.gz", bucket, model_key)
        print(f"Model uploaded to {s3_model_path}")
        
        # Create SageMaker model
        model_name = f"delivery-eta-backup-{int(time.time())}"
        
        # Create XGBoostModel in script mode using our inference.py
        sagemaker_session = sagemaker.Session()
        xgb_model = XGBoostModel(
            model_data=s3_model_path,
            role=role,
            entry_point="/tmp/inference.py",
            framework_version="1.7-1",
            py_version="py3",
            sagemaker_session=sagemaker_session
        )
        
        # Deploy endpoint
        endpoint_name = "delivery-eta-endpoint"
        existing_endpoints = sm.list_endpoints(NameContains=endpoint_name).get("Endpoints", [])

        def _get_status(name: str) -> str:
            try:
                return sm.describe_endpoint(EndpointName=name).get("EndpointStatus", "Unknown")
            except Exception:
                return "NotFound"

        if existing_endpoints:
            status = _get_status(endpoint_name)
            if status in ["Failed", "OutOfService"]:
                print(f"Existing endpoint is {status}. Deleting endpoint: {endpoint_name}")
                try:
                    sm.delete_endpoint(EndpointName=endpoint_name)
                except Exception as _:
                    pass
                # Wait until it's gone
                import time as _t
                for _ in range(60):
                    if _get_status(endpoint_name) == "NotFound":
                        break
                    _t.sleep(10)
                print(f"Creating endpoint: {endpoint_name}")
                xgb_model.deploy(
                    initial_instance_count=1,
                    instance_type="ml.t2.medium",
                    endpoint_name=endpoint_name
                )
            else:
                print(f"Updating endpoint: {endpoint_name}")
                xgb_model.deploy(
                    initial_instance_count=1,
                    instance_type="ml.t2.medium",
                    endpoint_name=endpoint_name
                )
        else:
            print(f"Creating endpoint: {endpoint_name}")
            xgb_model.deploy(
                initial_instance_count=1,
                instance_type="ml.t2.medium",
                endpoint_name=endpoint_name
            )
        
        print(f"Backup model deployed to SageMaker endpoint: {endpoint_name}")
        return True
        
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting backup model deployment...")
    deploy_local_model()