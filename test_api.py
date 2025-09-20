import boto3, json

runtime = boto3.client("sagemaker-runtime", region_name="ap-south-1")
endpoint = "delivery-eta-endpoint"

payload = {
    "product_weight_g": 500,
    "product_volume_cm3": 2000,
    "price": 49.99,
    "freight_value": 8.5,
    "purchase_hour": 14,
    "purchase_day_of_week": 3,
    "purchase_month": 6
}

# Convert to CSV (SageMaker expects text/csv for XGBoost)
csv_input = ",".join(str(payload[f]) for f in payload)

response = runtime.invoke_endpoint(
    EndpointName=endpoint,
    ContentType="text/csv",
    Body=csv_input
)

print("Prediction:", response["Body"].read().decode("utf-8"))
