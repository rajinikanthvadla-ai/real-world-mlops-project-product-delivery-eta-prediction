import pandas as pd, boto3, os, mlflow, mlflow.xgboost, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

MLFLOW_TRACKING_URI = "http://13.127.63.212:32001/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("delivery-eta-prediction")

s3 = boto3.client("s3")
bucket_processed = "product-delivery-eta-processed-data"
key_processed = "processed_data.csv"

local_path = "/tmp/processed_data.csv"

# Step 1: Try downloading processed data
try:
    s3.download_file(bucket_processed, key_processed, local_path)
    print("✅ Loaded processed data from S3")
    df = pd.read_csv(local_path)

except Exception as e:
    print("⚠️ Processed data not found in S3. Building from raw dataset...")
    # Download raw data from raw bucket
    raw_bucket = "product-delivery-eta-raw-data"
    local_dir = "/tmp/raw"
    os.makedirs(local_dir, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=raw_bucket):
        for obj in page.get("Contents", []):
            s3.download_file(raw_bucket, obj["Key"], os.path.join(local_dir, os.path.basename(obj["Key"])))

    # Load datasets
    orders_df = pd.read_csv(f"{local_dir}/olist_orders_dataset.csv")
    order_items_df = pd.read_csv(f"{local_dir}/olist_order_items_dataset.csv")
    products_df = pd.read_csv(f"{local_dir}/olist_products_dataset.csv")
    customers_df = pd.read_csv(f"{local_dir}/olist_customers_dataset.csv")
    sellers_df = pd.read_csv(f"{local_dir}/olist_sellers_dataset.csv")

    # Merge + process
    merged_df = order_items_df.merge(orders_df,on="order_id") \
                              .merge(products_df,on="product_id") \
                              .merge(customers_df,on="customer_id") \
                              .merge(sellers_df,on="seller_id")

    merged_df["order_purchase_timestamp"] = pd.to_datetime(merged_df["order_purchase_timestamp"], errors="coerce")
    merged_df["order_delivered_customer_date"] = pd.to_datetime(merged_df["order_delivered_customer_date"], errors="coerce")
    merged_df = merged_df.dropna(subset=["order_purchase_timestamp","order_delivered_customer_date"])

    merged_df["delivery_duration_days"] = (
        merged_df["order_delivered_customer_date"] - merged_df["order_purchase_timestamp"]
    ).dt.total_seconds() / (24*3600)

    merged_df = merged_df[(merged_df["delivery_duration_days"] > 0) & (merged_df["delivery_duration_days"] < 60)]
    merged_df["purchase_hour"] = merged_df["order_purchase_timestamp"].dt.hour
    merged_df["purchase_day_of_week"] = merged_df["order_purchase_timestamp"].dt.dayofweek
    merged_df["purchase_month"] = merged_df["order_purchase_timestamp"].dt.month
    merged_df["product_volume_cm3"] = merged_df["product_length_cm"] * merged_df["product_height_cm"] * merged_df["product_width_cm"]

    # Save + upload
    merged_df.to_csv(local_path,index=False)
    s3.upload_file(local_path, bucket_processed, key_processed)
    print("✅ Processed data built & uploaded to S3")
    df = merged_df

# Training
features = ['product_weight_g','product_volume_cm3','price','freight_value',
            'purchase_hour','purchase_day_of_week','purchase_month']
target = 'delivery_duration_days'
X,y = df[features], df[target]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

with mlflow.start_run() as run:
    params={"objective":"reg:squarederror","n_estimators":200,"max_depth":5,"learning_rate":0.1}
    model=xgb.XGBRegressor(**params).fit(X_train,y_train)

    preds=model.predict(X_test)
    rmse=mean_squared_error(y_test,preds,squared=False)
    mae=mean_absolute_error(y_test,preds)

    mlflow.log_params(params)
    mlflow.log_metrics({"rmse":rmse,"mae":mae})
    mlflow.xgboost.log_model(model,"model",registered_model_name="delivery-eta-model")

    print(f"✅ Run {run.info.run_id}: RMSE={rmse}, MAE={mae}")
