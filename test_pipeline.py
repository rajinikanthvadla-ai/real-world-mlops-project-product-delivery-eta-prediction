#!/usr/bin/env python3
"""
Quick test for GitHub Actions MLOps pipeline
Tests the MLflow connection and AWS credentials
"""

import os
import boto3
import mlflow
import sys

def test_mlflow_connection():
    """Test MLflow server connection"""
    try:
        mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://13.127.63.212:32001/')
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Try to connect and get server info
        client = mlflow.tracking.MlflowClient()
        
        # Test basic connectivity by searching experiments
        try:
            experiments = client.search_experiments()
            print(f"✅ MLflow connection successful to {mlflow_uri}")
            print(f"📊 Found {len(experiments)} experiments")
        except AttributeError:
            # Fallback for older MLflow versions
            experiments = client.list_experiments()
            print(f"✅ MLflow connection successful to {mlflow_uri}")
            print(f"📊 Found {len(experiments)} experiments")
        
        # Test creating a test run to verify full functionality
        with mlflow.start_run(run_name="connectivity_test") as run:
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 1.0)
            print(f"🔬 Test run created: {run.info.run_id}")
        
        return True
    except Exception as e:
        print(f"❌ MLflow connection failed: {str(e)}")
        print(f"🔍 MLflow URI: {mlflow_uri}")
        print(f"💡 Tip: Check if MLflow server is running and accessible")
        return False

def test_aws_credentials():
    """Test AWS credentials and permissions"""
    try:
        # Test basic AWS connectivity
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS connection successful")
        print(f"👤 Account: {identity['Account']}")
        print(f"🆔 User: {identity.get('Arn', 'Unknown')}")
        
        # Test S3 access
        s3 = boto3.client('s3')
        buckets = s3.list_buckets()
        print(f"📦 S3 access: Found {len(buckets['Buckets'])} buckets")
        
        # Test SageMaker access
        sm = boto3.client('sagemaker', region_name='ap-south-1')
        endpoints = sm.list_endpoints()
        print(f"🤖 SageMaker access: Found {len(endpoints['Endpoints'])} endpoints")
        
        return True
    except Exception as e:
        print(f"❌ AWS connection failed: {str(e)}")
        return False

def main():
    print("🧪 Testing MLOps Pipeline Prerequisites")
    print("=" * 50)
    
    # Test environment
    print(f"🌍 Environment: {'GitHub Actions' if 'GITHUB_ACTIONS' in os.environ else 'Local'}")
    print(f"🐍 Python: {sys.version}")
    
    # Run tests
    mlflow_ok = test_mlflow_connection()
    aws_ok = test_aws_credentials()
    
    print("\n" + "=" * 50)
    if aws_ok:  # AWS is critical, MLflow can be optional for some workflows
        if mlflow_ok:
            print("🎉 All tests passed! Pipeline should work correctly.")
        else:
            print("⚠️ MLflow connection failed, but AWS works. Pipeline may work with limited functionality.")
            print("💡 Consider running without MLflow tracking or fix MLflow server connection.")
        sys.exit(0)
    else:
        print("❌ AWS tests failed. Pipeline cannot work without AWS connectivity.")
        sys.exit(1)

if __name__ == "__main__":
    main()