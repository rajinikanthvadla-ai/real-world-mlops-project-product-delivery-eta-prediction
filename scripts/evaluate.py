import mlflow, pandas as pd, boto3
from sklearn.metrics import mean_squared_error

MLFLOW_TRACKING_URI = "http://13.127.63.212:32001/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

s3 = boto3.client("s3")
s3.download_file('product-delivery-eta-processed-data','processed_data.csv','/tmp/data.csv')
df = pd.read_csv('/tmp/data.csv')

features = ['product_weight_g','product_volume_cm3','price','freight_value',
            'purchase_hour','purchase_day_of_week','purchase_month']
X,y = df[features],df['delivery_duration_days']

prod = mlflow.pyfunc.load_model("models:/delivery-eta-model/Production")
staging = mlflow.pyfunc.load_model("models:/delivery-eta-model/Staging")

pr, sr = prod.predict(X), staging.predict(X)
prod_rmse = mean_squared_error(y,pr,squared=False)
stag_rmse = mean_squared_error(y,sr,squared=False)

print(f"Prod={prod_rmse}, Staging={stag_rmse}")
if stag_rmse < prod_rmse:
    ver = client.get_latest_versions("delivery-eta-model",["Staging"])[0].version
    client.transition_model_version_stage("delivery-eta-model",ver,"Production",archive_existing_versions=True)
    print(f"✅ Promoted staging {ver} → Production")
else:
    print("⚠️ Keeping current Production")
