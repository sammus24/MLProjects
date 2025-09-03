import json
import logging
import base64
import os
import joblib # For loading MinMaxScaler (from sklearn)
import pandas as pd # For DataFrame and data manipulation
import boto3 # For S3 client
import torch # For PyTorch model definition and loading
import torch.nn as nn # For PyTorch neural network modules
import numpy as np 
from datetime import datetime # For timestamp manipulation in feature engineering

logger = logging.getLogger()
logger.setLevel(logging.INFO) 

# --- S3 Configuration for Output ---
OUTPUT_BUCKET = 'sammu-fraud-detection-results' # <--- YOUR S3 BUCKET NAME!
S3_CLIENT = boto3.client('s3')


INPUT_DIM = 7 
HIDDEN_DIM = [64, 32, 16]
LATENT_DIM = 8

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder part: Reduce dimensions
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU(True))
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

        # Decoder part: Reconstruct original dimensions
        layers = []
        current_dim = latent_dim
        for h_dim in reversed(hidden_dim):
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU(True))
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

# --- Haversine Distance Function (MUST match train_model.py) ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# --- Feature Engineering Function for Inference (MUST match train_model.py) ---
def feature_engineer_data_for_inference(transaction_data, last_transaction_time_dict):
    """
    Applies advanced feature engineering to a single transaction data dictionary.
    Needs previous transaction time for velocity, and home_lat/lon from transaction_data.
    """
    required_raw_fields = ['timestamp', 'user_id', 'amount', 'latitude', 'longitude', 'home_lat', 'home_lon']
    if not all(field in transaction_data for field in required_raw_fields):
        logging.error(f"Missing raw fields for feature engineering: {required_raw_fields}. Transaction: {transaction_data.get('transaction_id', 'N/A')}")
        return None 

    try:
        current_timestamp = datetime.fromisoformat(transaction_data['timestamp'])
    except ValueError as e:
        logging.error(f"Timestamp parsing error for transaction {transaction_data.get('transaction_id', 'N/A')}: {e}")
        return None

    transaction_hour = current_timestamp.hour
    transaction_day_of_week = current_timestamp.dayofweek 

    distance_from_home = haversine_distance(
        transaction_data['latitude'], transaction_data['longitude'],
        transaction_data['home_lat'], transaction_data['home_lon']
    )
    
    user_id = transaction_data['user_id']
    time_since_last_transaction = 0 
    if user_id in last_transaction_time_dict:
        time_since_last_transaction = (current_timestamp - last_transaction_time_dict[user_id]).total_seconds()
    last_transaction_time_dict[user_id] = current_timestamp 
    
    engineered_features = {
        "amount": transaction_data['amount'],
        "transaction_hour": transaction_hour,
        "transaction_day_of_week": transaction_day_of_week,
        "distance_from_home": distance_from_home,
        "time_since_last_transaction": time_since_last_transaction,
        "latitude": transaction_data['latitude'],
        "longitude": transaction_data['longitude'],
    }

    # Handle potential infinite or NaN values from calculations, and ensure numeric types
    for key, value in engineered_features.items():
        if not isinstance(value, (int, float)): # Check if not int or float
            logging.warning(f"Feature '{key}' for transaction {transaction_data.get('transaction_id', 'N/A')} is not numeric: {value}. Imputing with 0.")
            engineered_features[key] = 0
        elif np.isinf(value) or np.isnan(value): # Check for inf/nan after confirming numeric
            engineered_features[key] = 0 
            
    return engineered_features

# --- Global variables for model and features ---
AUTOENCODER_MODEL_PATH = 'autoencoder_model.joblib' # Path for PyTorch model state_dict
SCALER_PATH = 'minmax_scaler.joblib' # Path for MinMaxScaler object


FEATURES = [
    "amount",
    "transaction_hour",
    "transaction_day_of_week",
    "distance_from_home",
    "time_since_last_transaction",
    "latitude",
    "longitude",
]

# Anomaly Threshold (MUST be copied from train_model.py output)
AUTOENCODER_ANOMALY_THRESHOLD = 0.0001

# --- Load model and scaler outside handler for efficiency (cold start) ---
ae_model = None
minmax_scaler = None
user_last_transaction_times = {} # This will retain state across warm invocations for velocity feature

try:
    # Load MinMaxScaler (saved with joblib)
    minmax_scaler = joblib.load(SCALER_PATH)
    logger.info("MinMaxScaler loaded successfully.")

    # Initialize Autoencoder model structure
    ae_model = Autoencoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
    # Load trained weights (saved with torch.save)
    # map_location='cpu' ensures it loads on CPU even if trained on GPU
    ae_model.load_state_dict(torch.load(AUTOENCODER_MODEL_PATH, map_location='cpu')) 
    ae_model.eval() # Set model to evaluation mode (disables dropout, batchnorm effects)
    logger.info("PyTorch Autoencoder model loaded successfully.")

except Exception as e:
    logger.error(f"Error loading model or scaler: {e}", exc_info=True)
    # Re-raise to force Lambda cold start failure if model loading is critical
    raise


def lambda_handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")

    if 'records' not in event:
        logger.warning("Event does not contain 'records' key. Skipping processing.")
        return {
            'statusCode': 200,
            'body': json.dumps('No Kafka records to process.')
        }

    processed_messages = 0
    anomalies_and_predictions = []

    # Access the global state for velocity feature (last_transaction_time_dict)
    global user_last_transaction_times 

    for topic_partition, records in event['records'].items():
        for record in records:
            try:
                decoded_value = base64.b64decode(record['value']).decode('utf-8')
                transaction_data = json.loads(decoded_value)
                
                logger.info(f"Processing transaction: [ID: {transaction_data.get('transaction_id', 'N/A')}, Amount: {transaction_data.get('amount', 'N/A')}, Is Fraud (GT): {transaction_data.get('is_fraud', 'N/A')}]")
                
                # --- Feature Engineering for Inference ---
                engineered_features_dict = feature_engineer_data_for_inference(transaction_data, user_last_transaction_times)
                if engineered_features_dict is None: # Feature engineering failed
                    logger.error(f"Feature engineering failed for transaction {transaction_data.get('transaction_id', 'N/A')}. Skipping inference.")
                    continue

               
                transaction_features_df = pd.DataFrame([engineered_features_dict])[FEATURES]

                # --- Real-Time Model Inference (Autoencoder) ---
                if ae_model and minmax_scaler:
                    # Scale features using the loaded scaler
                    scaled_features = minmax_scaler.transform(transaction_features_df)
                    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)

                    # Calculate reconstruction error
                    # No need to set ae_model.eval() repeatedly inside loop if it's already eval()
                    with torch.no_grad(): # Disable gradient calculation for inference
                        reconstruction = ae_model(input_tensor)
                        reconstruction_error = torch.mean((input_tensor - reconstruction)**2, dim=1).item() # .item() to get scalar
                    
                    # Classify based on threshold
                    is_predicted_anomaly = reconstruction_error > AUTOENCODER_ANOMALY_THRESHOLD 
                    
                    logger.info(f"  --> ML Prediction: Recon Error={reconstruction_error:.6f}, Threshold={AUTOENCODER_ANOMALY_THRESHOLD:.6f}, IsAnomaly={is_predicted_anomaly}")

                    # --- Store prediction results for S3 output ---
                    anomalies_and_predictions.append({
                        "transaction_id": transaction_data.get('transaction_id'),
                        "timestamp": transaction_data.get('timestamp'),
                        "amount": transaction_data.get('amount'),
                        "is_fraud_ground_truth": transaction_data.get('is_fraud'),
                        "reconstruction_error": float(reconstruction_error),
                        "anomaly_threshold": float(AUTOENCODER_ANOMALY_THRESHOLD),
                        "is_predicted_anomaly": is_predicted_anomaly
                    })
                else:
                    logger.warning("Autoencoder model or scaler not loaded. Skipping ML inference.")

                processed_messages += 1

            except Exception as e:
                logger.error(f"Error processing record: {record}. Error: {e}", exc_info=True)
    
    logger.info(f"Successfully processed {processed_messages} messages overall. Preparing to save results to S3.")

    # --- Save results to S3 ---
    if anomalies_and_predictions:
        output_filename = f"predictions/{context.aws_request_id}.json" # Unique filename per Lambda invocation
        try:
            output_content = "\n".join([json.dumps(item) for item in anomalies_and_predictions])
            S3_CLIENT.put_object(
                Bucket=OUTPUT_BUCKET,
                Key=output_filename,
                Body=output_content
            )
            logger.info(f"Saved {len(anomalies_and_predictions)} predictions to s3://{OUTPUT_BUCKET}/{output_filename}")
        except Exception as e:
            logger.error(f"Error saving predictions to S3: {e}", exc_info=True)
    else:
        logger.info("No predictions to save to S3 in this invocation.")

    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Successfully processed {processed_messages} messages.',
            'predictions_saved_to_s3': True if anomalies_and_predictions else False,
            's3_object_key': output_filename if anomalies_and_predictions else None
        })
    }
