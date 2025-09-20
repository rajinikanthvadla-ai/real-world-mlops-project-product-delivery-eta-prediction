import boto3, mlflow, os, time

MLFLOW_TRACKING_URI="http://13.127.63.212:32001/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client=mlflow.tracking.MlflowClient()

sm=boto3.client("sagemaker",region_name="ap-south-1")
account=boto3.client("sts").get_caller_identity()["Account"]
role=f"arn:aws:iam::{account}:role/service-role/AmazonSageMaker-ExecutionRole"
bucket="product-delivery-eta-model-artifacts"
endpoint="delivery-eta-endpoint"

# Export latest Production model
ver=client.get_latest_versions("delivery-eta-model",["Production"])[0]
local="/tmp/mlflow_model"
mlflow.pyfunc.download_artifacts(artifact_uri=ver.source,dst_path=local)

os.system(f"tar -czf /tmp/model.tar.gz -C {local} .")
boto3.client("s3").upload_file("/tmp/model.tar.gz",bucket,"mlflow_model.tar.gz")
model_url=f"s3://{bucket}/mlflow_model.tar.gz"

# Create Model + EndpointConfig
mname=f"eta-model-{int(time.time())}"
cname=f"{mname}-config"
sm.create_model(ModelName=mname,
    PrimaryContainer={"Image":"683313688378.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-xgboost:1.7-1","ModelDataUrl":model_url},
    ExecutionRoleArn=role)
sm.create_endpoint_config(EndpointConfigName=cname,
    ProductionVariants=[{"VariantName":"All","ModelName":mname,"InitialInstanceCount":1,"InstanceType":"ml.m5.large"}])

# Update or Create Endpoint
eps=sm.list_endpoints(NameContains=endpoint)["Endpoints"]
if eps: sm.update_endpoint(EndpointName=endpoint,EndpointConfigName=cname)
else: sm.create_endpoint(EndpointName=endpoint,EndpointConfigName=cname)

print(f"✅ Deployed Champion model to SageMaker Endpoint: {endpoint}")
