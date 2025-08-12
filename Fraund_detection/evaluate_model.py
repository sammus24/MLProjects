import pandas as pd
import boto3
import json
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import os
import io # For reading data from S3 objects in memory

# --- Configuration ---
S3_BUCKET_NAME = 'sammu-fraud-detection-results' # <--- YOUR S3 BUCKET NAME!
S3_PREFIX = 'predictions/' # Folder where Lambda saves prediction files

# Simulated Costs (Adjust these based on your hypothetical scenario)
COST_OF_TRUE_FRAUD_DETECTED = 1000 # Cost saved per true fraud detected
COST_OF_FALSE_POSITIVE = 100    # Cost incurred per legitimate transaction flagged as fraud (e.g., customer inconvenience, manual review)
AVERAGE_FRAUD_LOSS_PER_TRANSACTION = 500 # Average loss if a fraudulent transaction is *not* detected

# Initialize S3 client
s3 = boto3.client('s3')

def download_and_parse_predictions_from_s3(bucket_name, prefix):
    """Downloads all JSON prediction files from S3 and parses them."""
    print(f"Downloading predictions from s3://{bucket_name}/{prefix}...")
    all_predictions = []
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if key.endswith('.json'):
                        try:
                            response = s3.get_object(Bucket=bucket_name, Key=key)
                            # Read as string and then split by new line for JSONL format
                            content = response['Body'].read().decode('utf-8')
                            for line in content.splitlines():
                                if line.strip(): # Avoid empty lines
                                    all_predictions.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not decode JSON from {key}: {e}")
                        except Exception as e:
                            print(f"Error processing {key}: {e}")
    except Exception as e:
        print(f"Error listing objects in S3: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

    print(f"Downloaded and parsed {len(all_predictions)} prediction records.")
    if not all_predictions:
        print("No prediction data found. Please ensure your Lambda has run and saved files to S3.")
        return pd.DataFrame()

    df = pd.DataFrame(all_predictions)
    return df

def evaluate_model_performance(df):
    """Calculates intrinsic (F1, Precision, Recall) metrics."""
    print("\n--- Intrinsic Model Evaluation (F1-score, Precision, Recall) ---")
    
    y_pred = df['model_prediction_raw'].apply(lambda x: 1 if x == -1 else 0)
    # Ground truth: is_fraud_ground_truth (True/False) -> 1/0
    y_true = df['is_fraud_ground_truth'].astype(int) # Convert True/False to 1/0

    if y_true.empty or y_pred.empty:
        print("Insufficient data for evaluation.")
        return

   
    print(classification_report(y_true, y_pred, target_names=['Normal (0)', 'Fraud (1)']))

    # More specific metrics
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    print(f"Overall F1-Score (for Fraud/Positive class): {f1:.4f}")
    print(f"Overall Precision (for Fraud/Positive class): {precision:.4f}")
    print(f"Overall Recall (for Fraud/Positive class): {recall:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0) # Handle cases where only one class is present
    print("\nConfusion Matrix:")
    print(f"True Negatives (Correctly Normal): {tn}")
    print(f"False Positives (Legit flagged as Fraud): {fp}")
    print(f"False Negatives (Fraud missed): {fn}")
    print(f"True Positives (Fraud detected): {tp}")

    return tn, fp, fn, tp

def calculate_extrinsic_metrics(tn, fp, fn, tp, total_transactions_generated):
    """Calculates extrinsic (simulated cost savings, false positive reduction) metrics."""
    print("\n--- Extrinsic Model Evaluation (Simulated Business Impact) ---")

    # Total potential fraud if no system existed (assuming all 'is_fraud=True' would be a loss)
    total_potential_fraud_losses = (tp + fn) * AVERAGE_FRAUD_LOSS_PER_TRANSACTION
    
    # Cost of system operation / review
    cost_of_reviews = fp * COST_OF_FALSE_POSITIVE
    
    # Total savings from detected fraud
    savings_from_tp = tp * COST_OF_TRUE_FRAUD_DETECTED

    # Net simulated savings
    net_simulated_savings = savings_from_tp - cost_of_reviews
    
    print(f"Total potential fraud losses (if undetected): ${total_potential_fraud_losses:,.2f}")
    print(f"Savings from correctly detected fraud (True Positives): ${savings_from_tp:,.2f}")
    print(f"Cost from False Positives (manual review, customer inconvenience): ${cost_of_reviews:,.2f}")
    print(f"Net Simulated Cost Savings: ${net_simulated_savings:,.2f}")

    
    print(f"\nTotal False Positives (Legitimate transactions flagged as fraud): {fp}")
  
    print("\nNote: '85% reduction in false positives' is usually relative to a baseline system.")
    print("Our current False Positives (FP) count directly reflects the model's performance.")
    print("To meet the 85% reduction claim, you would adjust model parameters (e.g., contamination) and aim for a low FP count given your fraud rate.")

if __name__ == "__main__":
   
    predictions_df = download_and_parse_predictions_from_s3(S3_BUCKET_NAME, S3_PREFIX)

    if not predictions_df.empty:
        
        tn, fp, fn, tp = evaluate_model_performance(predictions_df)

        
        total_transactions = len(predictions_df)
        if total_transactions > 0: # Avoid division by zero
            calculate_extrinsic_metrics(tn, fp, fn, tp, total_transactions)
        else:
            print("No transactions to calculate extrinsic metrics.")
    else:
        print("No prediction data to evaluate.")