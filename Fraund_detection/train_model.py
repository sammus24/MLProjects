import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime

# --- Configuration ---
DATA_PATH = "simulated_transactions.csv"

# Model paths for Autoencoder ONLY
AUTOENCODER_MODEL_PATH = "autoencoder_model.joblib" # PyTorch model state_dict
SCALER_PATH = "minmax_scaler.joblib" # Scaler for Autoencoder

# Advanced Features - MUST align with features used in Lambda
ADVANCED_FEATURES = [
    "amount",
    "transaction_hour",
    "transaction_day_of_week",
    "distance_from_home",
    "time_since_last_transaction",
    "latitude",
    "longitude",
]

# Autoencoder Hyperparameters (MUST match values used for training)
INPUT_DIM = len(ADVANCED_FEATURES)
HIDDEN_DIM = [64, 32, 16]
LATENT_DIM = 8
NUM_EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# --- Autoencoder Model Definition (MUST match lambda_function.py) ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder part
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU(True))
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

        # Decoder part
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

# --- Haversine Distance Function (MUST match lambda_function.py) ---
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

# --- Feature Engineering Function (MUST match lambda_function.py) ---
def feature_engineer_data(df):
    print("Performing advanced feature engineering...")
    
    required_cols = ['timestamp', 'user_id', 'latitude', 'longitude', 'home_lat', 'home_lon']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: Missing required columns for feature engineering: {missing}. "
              "Please ensure your datageneration.py includes these in CSV output.")
        exit()

    df['transaction_hour'] = df['timestamp'].dt.hour
    df['transaction_day_of_week'] = df['timestamp'].dt.dayofweek 

    df['distance_from_home'] = df.apply(
        lambda row: haversine_distance(row['latitude'], row['longitude'], row['home_lat'], row['home_lon']),
        axis=1
    )
    df['distance_from_home'] = df['distance_from_home'].fillna(0)

    df['time_since_last_transaction'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(0)
    
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0) 

    print("Feature engineering complete.")
    return df

# --- Data Loading and Preparation for ML Models ---
# This will still perform the split, but we'll only use X_normal_train_df for AE training
def load_and_prepare_data_for_ae_training(file_path):
    print(f"\n--- Loading and preparing data from {file_path} ---")
    
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    print(f"Initial loaded {len(df)} transactions.")
    print(f"Initial 'is_fraud=True' labels: {df['is_fraud'].sum()}")

    processed_df = feature_engineer_data(df.copy())

    normal_data = processed_df[processed_df['is_fraud'] == False].copy()
    fraud_data = processed_df[processed_df['is_fraud'] == True].copy()
    
    print(f"Separated data: {len(normal_data)} normal, {len(fraud_data)} fraudulent.")

    # Split normal data into training and validation sets for Autoencoder
    X_normal_train, X_normal_test = train_test_split(
        normal_data, test_size=0.2, random_state=42, 
        stratify=normal_data['user_id'] if 'user_id' in normal_data.columns else None
    )
    # We don't explicitly need X_fraud_test here for training only
    
    print(f"Data Split Summary: {len(X_normal_train)} normal samples for training (AE)")

    return X_normal_train, X_normal_test # Returning normal test for potential threshold calculation after training

# --- Model Training Function (Autoencoder only) ---
def train_autoencoder_model(dataframe, features, model_path, scaler_path, input_dim, hidden_dim, latent_dim, num_epochs, batch_size, learning_rate):
    """Trains and saves a PyTorch Autoencoder model."""
    print(f"\n--- Training PyTorch Autoencoder on {len(dataframe)} normal samples ---")
    X_train_raw = dataframe[features]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)

    model = Autoencoder(input_dim, hidden_dim, latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for data in train_loader:
            inputs = data[0]
            reconstructions = model(inputs)
            loss = criterion(reconstructions, inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    print("Autoencoder training complete.")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Autoencoder model saved to {model_path}")
    print(f"MinMaxScaler saved to {scaler_path}")

    return model, scaler # Return for potential immediate evaluation / threshold calculation


if __name__ == "__main__":
    # --- Load and Prepare Data for AE Training ---
    X_normal_train_df, X_normal_test_df = load_and_prepare_data_for_ae_training(DATA_PATH)
    
    # --- Train PyTorch Autoencoder ---
    print("\n=== Training PyTorch Autoencoder ===")
    ae_model, minmax_scaler = train_autoencoder_model(
        X_normal_train_df, ADVANCED_FEATURES, AUTOENCODER_MODEL_PATH, SCALER_PATH, 
        INPUT_DIM, HIDDEN_DIM, LATENT_DIM, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE
    )

    print("\nAutoencoder model training finished.")
    print(f"Model trained on features: {ADVANCED_FEATURES}")

    # --- Calculate and Print Anomaly Threshold (for use in lambda_function.py) ---
    print("\n=== Calculating Anomaly Threshold ===")
    ae_model.eval() # Set model to evaluation mode
    
    # Prepare normal test data for threshold calculation
    X_normal_test_features_scaled = minmax_scaler.transform(X_normal_test_df[ADVANCED_FEATURES])
    X_normal_test_tensor = torch.tensor(X_normal_test_features_scaled, dtype=torch.float32)
    with torch.no_grad():
        normal_reconstructions = ae_model(X_normal_test_tensor)
        normal_reconstruction_errors = torch.mean((X_normal_test_tensor - normal_reconstructions)**2, dim=1).numpy()
    
    AUTOENCODER_ANOMALY_THRESHOLD = np.percentile(normal_reconstruction_errors, 99.5) 
    print(f"Autoencoder Anomaly Threshold (99.5 percentile of normal test data errors): {AUTOENCODER_ANOMALY_THRESHOLD:.4f}")
    print("\nThis threshold value should be copied into your lambda_function.py and used for real-time anomaly classification.")

    print("\nAll training steps completed. Next: Update lambda_function.py, rebuild Docker image, push to ECR, and update Lambda function.")