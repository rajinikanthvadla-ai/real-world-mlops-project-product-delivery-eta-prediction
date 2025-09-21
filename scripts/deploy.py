import boto3, mlflow, os, time, json, tarfile
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.xgboost.model import XGBoostModel
import numpy as np
import pandas as pd

MLFLOW_TRACKING_URI="http://13.203.199.220:32001/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client=mlflow.tracking.MlflowClient()

sm=boto3.client("sagemaker",region_name="ap-south-1")
sagemaker_session = sagemaker.Session()

def _resolve_sagemaker_role() -> str:
    env_role = os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN")
    if env_role:
        print(f"Using IAM role from env: {env_role}")
        return env_role
    try:
        return sagemaker.get_execution_role()
    except Exception:
        pass
    try:
        iam = boto3.client("iam")
        marker = None
        while True:
            kwargs = {"Marker": marker} if marker else {}
            resp = iam.list_roles(**kwargs)
            for r in resp.get("Roles", []):
                if r.get("RoleName", "").startswith("AmazonSageMaker-ExecutionRole"):
                    print(f"Using discovered IAM role: {r['Arn']}")
                    return r["Arn"]
            if resp.get("IsTruncated"):
                marker = resp.get("Marker")
            else:
                break
    except Exception:
        pass
    account = boto3.client("sts").get_caller_identity()["Account"]
    fallback = f"arn:aws:iam::{account}:role/service-role/AmazonSageMaker-ExecutionRole"
    print(f"Using fallback IAM role: {fallback}")
    return fallback

role = _resolve_sagemaker_role()
bucket = sagemaker.Session().default_bucket()
endpoint_name="delivery-eta-endpoint"

# Create a proper inference script
def create_inference_script():
    inference_code = '''
import os
import xgboost as xgb
import numpy as np
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES = ['product_weight_g','product_volume_cm3','price','freight_value','purchase_hour','purchase_day_of_week','purchase_month']

def model_fn(model_dir):
    """Load the model from the model_dir directory"""
    logger.info(f"Loading model from: {model_dir}")
    
    # Check for different model file formats
    model_files = [
        os.path.join(model_dir, 'model.json'),
        os.path.join(model_dir, 'model.bst'),
        os.path.join(model_dir, 'model.pkl'),
        os.path.join(model_dir, 'model.joblib'),
        os.path.join(model_dir, 'xgboost-model'),
        os.path.join(model_dir, 'xgb-model.json'),
        os.path.join(model_dir, 'xgb-model')
    ]
    
    model = None
    loaded_model_path = None
    
    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                logger.info(f"Attempting to load model from: {model_path}")
                
                if model_path.endswith('.json') or model_path.endswith('.bst'):
                    model = xgb.Booster()
                    model.load_model(model_path)
                    logger.info(f"Successfully loaded XGBoost model from: {model_path}")
                elif model_path.endswith('.pkl') or model_path.endswith('.joblib'):
                    import joblib
                    model = joblib.load(model_path)
                    logger.info(f"Successfully loaded joblib model from: {model_path}")
                
                loaded_model_path = model_path
                break
                
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {str(e)}")
                continue
    
    if model is None:
        # Check what files are actually available
        available_files = []
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                available_files.append(os.path.join(root, file))
        
        error_msg = f"No supported model file found under {model_dir}. Available files: {available_files}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Model loaded successfully from: {loaded_model_path}")
    return model

def input_fn(request_body, request_content_type):
    """Parse input data"""
    logger.info(f"Content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        if isinstance(data, dict) and 'instances' in data:
            # TF serving format
            return np.array(data['instances'], dtype=np.float32)
        else:
            # Direct array format
            return np.array(data, dtype=np.float32)
    elif request_content_type == 'text/csv':
        # Parse CSV
        values = [float(x.strip()) for x in request_body.split(',')]
        return np.array(values, dtype=np.float32).reshape(1, -1)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    if isinstance(model, xgb.Booster):
        dmatrix = xgb.DMatrix(input_data, feature_names=FEATURES)
        return model.predict(dmatrix)
    else:
        return model.predict(input_data)

def output_fn(prediction, accept):
    """Format output"""
    if accept == 'application/json':
        return json.dumps({'predictions': prediction.tolist()})
    else:
        # Default to CSV
        return ','.join(str(x) for x in prediction)
'''
    
    # Write inference script to a temporary file
    inference_script_path = '/tmp/inference.py'
    with open(inference_script_path, 'w') as f:
        f.write(inference_code)
    
    return inference_script_path

def deploy_production_model():
    """Deploy the Production model from MLflow to SageMaker"""
    try:
        # Check if model registry exists
        try:
            # Get Production model from MLflow
            prod_versions = client.get_latest_versions("delivery-eta-model", ["Production"])
        except Exception as registry_error:
            print(f"Model registry not accessible: {registry_error}")
            print("No models in registry. Use backup deployment instead.")
            return False
            
        if not prod_versions:
            print("No Production model found. Checking for Staging model...")
            staging_versions = client.get_latest_versions("delivery-eta-model", ["Staging"])
            if not staging_versions:
                print("No models found in Staging or Production")
                print("Use backup deployment with local model instead.")
                return False
            model_version = staging_versions[0]
            # Promote to Production
            client.transition_model_version_stage(
                "delivery-eta-model", 
                model_version.version, 
                "Production"
            )
            print(f"Promoted Staging v{model_version.version} to Production")
        else:
            model_version = prod_versions[0]
        
        print(f"Deploying model version {model_version.version} to SageMaker...")
        
        # Download model artifacts
        local_path = "/tmp/mlflow_model"
        os.makedirs(local_path, exist_ok=True)
        mlflow.artifacts.download_artifacts(
            artifact_uri=model_version.source,
            dst_path=local_path
        )
        
        # Check what files were downloaded
        print(f"Downloaded files: {os.listdir(local_path)}")
        for root, dirs, files in os.walk(local_path):
            for file in files:
                print(f"Found file: {os.path.join(root, file)}")
        
        # Create model tarball with proper structure
        model_tar_path = "/tmp/model.tar.gz"
        with tarfile.open(model_tar_path, "w:gz") as tar:
            tar.add(local_path, arcname=".")
        
        # Upload to S3
        s3_model_path = f"s3://{bucket}/models/delivery-eta-v{model_version.version}/model.tar.gz"
        s3_client = boto3.client("s3")
        s3_client.upload_file(model_tar_path, bucket, f"models/delivery-eta-v{model_version.version}/model.tar.gz")
        
        # Create inference script
        inference_script_path = create_inference_script()
        
        # Create XGBoost model
        skl_model = XGBoostModel(
            model_data=s3_model_path,
            role=role,
            entry_point=inference_script_path,
            framework_version='1.7-1',
            py_version='py3',
            sagemaker_session=sagemaker_session,
            code_location=f"s3://{bucket}/code/",
            source_dir=None  # We're using a direct file path
        )
        
        # Check if endpoint exists and its status
        def _get_status(name: str) -> str:
            try:
                return sm.describe_endpoint(EndpointName=name).get("EndpointStatus", "Unknown")
            except Exception:
                return "NotFound"
        
        status = _get_status(endpoint_name)
        
        if status == "NotFound":
            # Create new endpoint
            print(f"Creating new endpoint: {endpoint_name}")
            unique_cfg = f"{endpoint_name}-cfg-{int(time.time())}"
            predictor = skl_model.deploy(
                initial_instance_count=1,
                instance_type='ml.m5.large',
                endpoint_name=endpoint_name,
                endpoint_config_name=unique_cfg,
                wait=True
            )
        else:
            # Update existing endpoint
            print(f"Updating existing endpoint: {endpoint_name}")
            
            # First create a new endpoint config
            unique_cfg = f"{endpoint_name}-cfg-{int(time.time())}"
            
            # Create the endpoint config
            skl_model.create_predictor(endpoint_name=endpoint_name).create_endpoint_config(
                name=unique_cfg,
                initial_instance_count=1,
                instance_type='ml.m5.large'
            )
            
            # Update the endpoint
            sm.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=unique_cfg
            )
            
            # Wait for update to complete
            waiter = sm.get_waiter('endpoint_in_service')
            waiter.wait(EndpointName=endpoint_name)
        
        print(f"Model v{model_version.version} deployed to SageMaker endpoint: {endpoint_name}")
        return True
        
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = deploy_production_model()
    if not success:
        print("MLflow deployment failed - use backup deployment")
        exit(1)
