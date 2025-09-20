import boto3
import json
import pandas as pd

# Test data for prediction
test_data = {
    "product_weight_g": 500,
    "product_volume_cm3": 1000,
    "price": 29.99,
    "freight_value": 8.50,
    "purchase_hour": 14,
    "purchase_day_of_week": 2,
    "purchase_month": 6
}

def test_sagemaker_endpoint():
    """Test the deployed SageMaker endpoint"""
    
    endpoint_name = "delivery-eta-endpoint"
    
    try:
        # Check if endpoint exists before invoking
        sm = boto3.client('sagemaker', region_name='ap-south-1')
        endpoints = sm.list_endpoints(NameContains=endpoint_name).get("Endpoints", [])
        if not endpoints:
            print(f"Endpoint {endpoint_name} not found. Skipping endpoint test.")
            return False
        # Create SageMaker runtime client
        runtime = boto3.client('sagemaker-runtime', region_name='ap-south-1')
        
        # XGBoost container expects CSV format
        features = ['product_weight_g','product_volume_cm3','price','freight_value',
                   'purchase_hour','purchase_day_of_week','purchase_month']
        csv_input = ",".join(str(test_data[f]) for f in features)
        
        print(f"Test Data: {test_data}")
        print(f"Invoking endpoint: {endpoint_name} (CSV)")
        print(f"CSV Input: {csv_input}")
        
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=csv_input
        )
        
        result = response["Body"].read().decode("utf-8").strip()
        print(f"CSV prediction successful!")
        print(f"Predicted delivery time: {float(result):.2f} days")
        
        return True
        
    except Exception as e:
        print(f"Endpoint test failed: {str(e)}")
        return False

def test_local_model():
    """Test locally saved model"""
    try:
        import joblib
        import os
        
        model_path = "./models/latest_model.joblib"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            # Create test DataFrame
            test_df = pd.DataFrame([test_data])
            
            prediction = model.predict(test_df)
            
            print(f"Local model test successful!")
            print(f"Local prediction: {prediction[0]:.2f} days")
            return True
        else:
            print(f"No local model found at {model_path}")
            return False
            
    except Exception as e:
        print(f"Local model test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Model Predictions")
    print("=" * 40)
    
    # Test local model first
    local_ok = test_local_model()
    print()
    
    # Test SageMaker endpoint
    endpoint_ok = test_sagemaker_endpoint()
    
    print("\n" + "=" * 40)
    if local_ok or endpoint_ok:
        print("Model testing completed successfully!")
    else:
        print("All model tests failed")
