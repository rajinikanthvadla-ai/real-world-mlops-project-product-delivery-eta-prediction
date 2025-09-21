import pandas as pd, boto3, mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

mlflow.set_tracking_uri("http://13.203.199.220:32001/")
s3=boto3.client("s3")
s3.download_file('product-delivery-eta-processed-data','processed_data.csv','/tmp/ref.csv')
ref=pd.read_csv('/tmp/ref.csv')
cur=ref.sample(frac=0.1)

report=Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref,current_data=cur)
res=report.as_dict()

if res['metrics'][0]['result']['dataset_drift']:
    print("⚠️ Drift detected:",res['metrics'][0]['result']['drifted_columns'])
else:
    print("✅ No drift")
