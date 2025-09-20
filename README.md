# Product Delivery ETA Prediction (MLOps Project)

This project demonstrates a **Champion–Challenger MLOps workflow** on AWS:
- MLflow on EKS for experiment tracking
- S3 + RDS for storage
- SageMaker for deployment
- GitHub Actions for CI/CD
- Evidently for drift monitoring

## Workflows
- **cicd.yml**: Train → Evaluate → Deploy to SageMaker
- **monitor.yml**: Daily drift detection

## Setup
1. Configure GitHub secrets:
   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_REGION=ap-south-1
2. Trigger pipeline manually via GitHub Actions
3. Use `test_api.py` to query the SageMaker endpoint

## Endpoint
- Name: `delivery-eta-endpoint`
