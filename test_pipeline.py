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
        
        # Try to list experiments
        client = mlflow.tracking.MlflowClient()
        experiments = client.list_experiments()
        
        print(f"✅ MLflow connection successful to {mlflow_uri}")
        print(f"📊 Found {len(experiments)} experiments")
        return True
    except Exception as e:
        print(f"❌ MLflow connection failed: {str(e)}")
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
    if mlflow_ok and aws_ok:
        print("🎉 All tests passed! Pipeline should work correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check your configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()