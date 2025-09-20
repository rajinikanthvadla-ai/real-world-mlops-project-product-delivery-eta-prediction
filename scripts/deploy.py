import boto3, mlflow, os, time
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor

MLFLOW_TRACKING_URI="http://13.127.63.212:32001/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client=mlflow.tracking.MlflowClient()

sm=boto3.client("sagemaker",region_name="ap-south-1")
sagemaker_session = sagemaker.Session()

# Handle role detection for different environments
try:
    role = sagemaker.get_execution_role()  # Works in SageMaker
except ValueError:
    # GitHub Actions or local environment
    account = boto3.client("sts").get_caller_identity()["Account"]
    role = f"arn:aws:iam::{account}:role/service-role/AmazonSageMaker-ExecutionRole"
    print(f"Using IAM role: {role}")
bucket="product-delivery-eta-model-artifacts"
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
        
        # Create SageMaker model
        model_name = f"delivery-eta-model-v{model_version.version}-{int(time.time())}"
        # Resolve correct XGBoost inference image for the region
        container_image = sagemaker.image_uris.retrieve(framework="xgboost", region="ap-south-1", version="1.7-1")
        
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": container_image,
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
                    "InstanceType": "ml.t2.medium",  # Cost-effective for demo
                    "InitialVariantWeight": 1.0
                }
            ]
        )
        
        # Deploy or update endpoint
        existing_endpoints = sm.list_endpoints(NameContains=endpoint_name)["Endpoints"]
        
        if existing_endpoints:
            print(f"Updating existing endpoint: {endpoint_name}")
            sm.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
        else:
            print(f"Creating new endpoint: {endpoint_name}")
            sm.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
        
        # Wait for deployment
        print("Waiting for endpoint deployment...")
        waiter = sm.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        
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
