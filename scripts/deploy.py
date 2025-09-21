import boto3, mlflow, os, time
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.xgboost.model import XGBoostModel

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
# Use SageMaker default bucket to ensure access from the role
bucket = sagemaker.Session().default_bucket()
endpoint_name="delivery-eta-endpoint"

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
        
        # Create model tarball
        os.system(f"tar -czf /tmp/model.tar.gz -C {local_path} .")
        
        # Upload to S3
        s3_model_path = f"s3://{bucket}/models/delivery-eta-v{model_version.version}/model.tar.gz"
        boto3.client("s3").upload_file("/tmp/model.tar.gz", bucket, f"models/delivery-eta-v{model_version.version}/model.tar.gz")
        
        # Use script-mode XGBoostModel with a minimal inference.py
        # Create an inference script in /tmp and include with model by pointing entry_point
        infer_code = '''
import os
import xgboost as xgb
import numpy as np

FEATURES = ['product_weight_g','product_volume_cm3','price','freight_value','purchase_hour','purchase_day_of_week','purchase_month']

def _find_model_file(model_dir):
    for root, _, files in os.walk(model_dir):
        # Prefer native XGBoost booster
        if 'xgboost-model' in files:
            return os.path.join(root, 'xgboost-model'), 'booster'
        # Common joblib/pickle names
        for name in ['model.joblib', 'model.pkl', 'model.bin']:
            if name in files:
                return os.path.join(root, name), 'joblib'
    return None, None

def model_fn(model_dir):
    path, kind = _find_model_file(model_dir)
    if path is None:
        raise FileNotFoundError(f'No supported model file found under {model_dir}')
    if kind == 'booster':
        booster = xgb.Booster()
        booster.load_model(path)
        return booster
    import joblib
    return joblib.load(path)

def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        values = [float(x.strip()) for x in request_body.split(',')]
        try:
            import pandas as pd
            return pd.DataFrame([values], columns=FEATURES)
        except Exception:
            return np.array(values, dtype=np.float32).reshape(1, -1)
    raise ValueError(f'Unsupported content type: {request_content_type}')

def predict_fn(data, model):
    if isinstance(model, xgb.Booster):
        dmatrix = xgb.DMatrix(data, feature_names=FEATURES)
        return model.predict(dmatrix)
    return model.predict(data)

def output_fn(prediction, content_type):
    return str(float(prediction[0]))
'''
        with open('/tmp/inference.py', 'w') as f:
            f.write(infer_code)

        sagemaker_session = sagemaker.Session()
        skl_model = XGBoostModel(
            model_data=s3_model_path,
            role=role,
            entry_point='/tmp/inference.py',
            framework_version='1.7-1',
            py_version='py3',
            sagemaker_session=sagemaker_session
        )
        
        # Deploy or update endpoint
        existing_endpoints = sm.list_endpoints(NameContains=endpoint_name).get("Endpoints", [])

        # Ensure stale default-named endpoint-config is removed to avoid name conflict
        try:
            sm.describe_endpoint_config(EndpointConfigName=endpoint_name)
            print(f"Found stale endpoint-config {endpoint_name}, deleting it before deploy...")
            sm.delete_endpoint_config(EndpointConfigName=endpoint_name)
        except Exception:
            pass

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
                except Exception:
                    pass
                # Wait until deleted
                import time as _t
                for _ in range(60):
                    if _get_status(endpoint_name) == "NotFound":
                        break
                    _t.sleep(10)
                print(f"Creating new endpoint: {endpoint_name}")
                unique_cfg = f"{endpoint_name}-cfg-{int(time.time())}"
                skl_model.deploy(
                    initial_instance_count=1,
                    instance_type='ml.m5.large',
                    endpoint_name=endpoint_name,
                    endpoint_config_name=unique_cfg
                )
            else:
                print(f"Updating existing endpoint: {endpoint_name}")
                unique_cfg = f"{endpoint_name}-cfg-{int(time.time())}"
                skl_model.deploy(
                    initial_instance_count=1,
                    instance_type='ml.m5.large',
                    endpoint_name=endpoint_name,
                    endpoint_config_name=unique_cfg
                )
        else:
            print(f"Creating new endpoint: {endpoint_name}")
            unique_cfg = f"{endpoint_name}-cfg-{int(time.time())}"
            skl_model.deploy(
                initial_instance_count=1,
                instance_type='ml.m5.large',
                endpoint_name=endpoint_name,
                endpoint_config_name=unique_cfg
            )
        
        # Wait for deployment
        print(f"Model v{model_version.version} deployed to SageMaker endpoint: {endpoint_name}")
        return True
        
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = deploy_production_model()
    if not success:
        print("MLflow deployment failed - use backup deployment")
        exit(1)
