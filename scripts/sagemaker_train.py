"""
SageMaker Training Job Launcher
This script launches training jobs on SageMaker and integrates with MLflow
"""
import boto3
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker import get_execution_role
import time

def launch_sagemaker_training():
    """Launch training job on SageMaker with MLflow tracking"""
    
    # SageMaker setup
    sagemaker_session = sagemaker.Session()
    role = get_execution_role()  # Or specify your role ARN
    
    # Training configuration
    training_instance_type = "ml.m5.large"
    training_instance_count = 1
    
    # Create SKLearn estimator (XGBoost will be installed via requirements)
    estimator = SKLearn(
        entry_point='train.py',
        source_dir='scripts',
        role=role,
        instance_type=training_instance_type,
        instance_count=training_instance_count,
        framework_version='1.0-1',
        py_version='py3',
        script_mode=True,
        hyperparameters={
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1
        },
        # Enable MLflow tracking
        environment={
            'MLFLOW_TRACKING_URI': 'http://13.203.199.220:32001/',
            'MLFLOW_EXPERIMENT_NAME': 'sagemaker-delivery-eta-prediction'
        }
    )
    
    # Launch training job
    job_name = f"delivery-eta-training-{int(time.time())}"
    print(f"🚀 Launching SageMaker training job: {job_name}")
    
    estimator.fit(job_name=job_name)
    
    print(f"✅ Training job completed: {job_name}")
    print(f"📊 Model artifacts: {estimator.model_data}")
    
    return estimator

if __name__ == "__main__":
    estimator = launch_sagemaker_training()