
import pandas as pd, boto3, mlflow, mlflow.xgboost, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

MLFLOW_TRACKING_URI = "http://13.127.63.212:32001/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("delivery-eta-prediction")

s3 = boto3.client("s3")
s3.download_file('product-delivery-eta-processed-data','processed_data.csv','/tmp/processed_data.csv')
df = pd.read_csv('/tmp/processed_data.csv')

features = ['product_weight_g','product_volume_cm3','price','freight_value',
            'purchase_hour','purchase_day_of_week','purchase_month']
target = 'delivery_duration_days'
X, y = df[features], df[target]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

with mlflow.start_run() as run:
    params = {"objective":"reg:squarederror","n_estimators":200,"max_depth":5,"learning_rate":0.1}
    model = xgb.XGBRegressor(**params).fit(X_train,y_train)
    preds = model.predict(X_test)

    mlflow.log_params(params)
    mlflow.log_metrics({
        "rmse": mean_squared_error(y_test,preds,squared=False),
        "mae": mean_absolute_error(y_test,preds)
    })
    mlflow.xgboost.log_model(model,"model",registered_model_name="delivery-eta-model")
