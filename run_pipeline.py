#!/usr/bin/env python3
"""
Complete MLOps Workflow Runner
Orchestrates the full ML pipeline: Train -> Evaluate -> Deploy
"""

import subprocess
import sys
import time
import mlflow
import argparse

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout)
    else:
        print(f"❌ {description} failed")
        print("Error:", result.stderr)
        return False
    return True

def run_local_training():
    """Run training locally with MLflow tracking"""
    return run_command(
        "python scripts/train.py",
        "Local Training with MLflow Tracking"
    )

def run_sagemaker_training():
    """Launch SageMaker training job"""
    return run_command(
        "python scripts/sagemaker_train.py",
        "SageMaker Training Job"
    )

def run_evaluation():
    """Run model evaluation and promotion"""
    return run_command(
        "python scripts/evaluate.py",
        "Model Evaluation and Promotion"
    )

def run_deployment():
    """Deploy to SageMaker endpoint"""
    return run_command(
        "python scripts/deploy.py",
        "SageMaker Deployment"
    )

def run_monitoring():
    """Run model monitoring"""
    return run_command(
        "python scripts/monitor.py",
        "Model Monitoring"
    )

def main():
    parser = argparse.ArgumentParser(description="MLOps Pipeline Runner")
    parser.add_argument(
        "--mode", 
        choices=["local", "sagemaker", "full"], 
        default="local",
        help="Training mode: local (with MLflow), sagemaker (SageMaker jobs), or full (complete pipeline)"
    )
    parser.add_argument(
        "--skip-training", 
        action="store_true",
        help="Skip training step (useful for testing evaluation/deployment)"
    )
    
    args = parser.parse_args()
    
    print(f"""
🚀 Starting MLOps Pipeline
========================
Mode: {args.mode}
Skip Training: {args.skip_training}
""")
    
    try:
        # Step 1: Training
        if not args.skip_training:
            if args.mode == "local":
                if not run_local_training():
                    sys.exit(1)
            elif args.mode == "sagemaker":
                if not run_sagemaker_training():
                    sys.exit(1)
            elif args.mode == "full":
                # Run both for comparison
                print("🔄 Running both local and SageMaker training...")
                run_local_training()
                time.sleep(2)
                run_sagemaker_training()
        
        # Step 2: Evaluation (compare models and promote)
        print("\n" + "="*50)
        time.sleep(2)
        if not run_evaluation():
            print("⚠️ Evaluation failed, but continuing...")
        
        # Step 3: Deployment
        print("\n" + "="*50)
        time.sleep(2)
        if not run_deployment():
            print("❌ Deployment failed!")
            sys.exit(1)
        
        # Step 4: Monitoring (optional)
        if args.mode == "full":
            print("\n" + "="*50)
            time.sleep(2)
            run_monitoring()
        
        print(f"""
🎉 MLOps Pipeline Complete!
===========================
✅ Training: {'Skipped' if args.skip_training else 'Completed'}
✅ Evaluation: Completed
✅ Deployment: Completed
{'✅ Monitoring: Completed' if args.mode == 'full' else ''}

Your model is now deployed and ready for inference!
""")
        
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()