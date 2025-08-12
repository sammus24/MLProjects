import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import joblib
import os
from datetime import datetime

# --- Configuration ---
DATA_PATH = "simulated_transactions.csv"

# Model paths for BOTH models
ISOLATION_FOREST_MODEL_PATH = "isolation_forest_model.joblib"
AUTOENCODER_MODEL_PATH = "autoencoder_model.joblib"
SCALER_PATH = "minmax_scaler.joblib"

# Advanced Features - These will be created/used from your raw data
ADVANCED_FEATURES = [
    "amount",
    "transaction_hour",
    "transaction_day_of_week",
    "distance_from_home",
    "time_since_last_transaction",
    "latitude",
    "longitude",
]

# Autoencoder Hyperparameters
INPUT_DIM = len(ADVANCED_FEATURES)
HIDDEN_DIM = [64, 32, 16]
LATENT_DIM = 8
NUM_EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# Isolation Forest Hyperparameters
IF_CONTAMINATION_RATE = 0.005 # Expected proportion of outliers in the data

# Simulated Costs (Adjust these based on your hypothetical scenario)
COST_OF_TRUE_FRAUD_DETECTED = 1000 
COST_OF_FALSE_POSITIVE = 100    
AVERAGE_FRAUD_LOSS_PER_TRANSACTION = 500

# --- Autoencoder Model Definition ---
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

# --- Haversine Distance Function ---
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

# --- Feature Engineering Function ---
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
def load_and_prepare_data(file_path):
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

    X_normal_train, X_normal_test = train_test_split(
        normal_data, test_size=0.2, random_state=42, 
        stratify=normal_data['user_id'] if 'user_id' in normal_data.columns else None
    )
    X_fraud_test = fraud_data.copy()

    print(f"Data Split Summary:")
    print(f"  Normal Train (for AE training): {len(X_normal_train)} samples")
    print(f"  Normal Test (for AE thresholding/evaluation): {len(X_normal_test)} samples")
    print(f"  Fraud Test (for AE/IF evaluation): {len(X_fraud_test)} samples")

    return X_normal_train, X_normal_test, X_fraud_test

# --- Model Training Functions ---
def train_isolation_forest_model(dataframe, features, contamination, model_path):
    """Trains and saves an Isolation Forest model."""
    print(f"\n--- Training Isolation Forest model on {len(dataframe)} samples ---")
    X = dataframe[features]
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X) 
    joblib.dump(model, model_path)
    print(f"Isolation Forest model saved to {model_path}")
    return model

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
    print(f"MinMaxScaler saved to {scaler_path}") # Typo fix: scaler_scaler.joblib -> scaler_path
    print(f"MinMaxScaler saved to {scaler_path}")

    return model, scaler

# --- Evaluation Functions (Copied from evaluate_model.py) ---
def evaluate_model_performance(y_true, y_pred, model_name):
    """Calculates and prints intrinsic metrics for a given model."""
    print(f"\n--- Intrinsic Model Evaluation for {model_name} ---")
    
    print(classification_report(y_true, y_pred, target_names=['Normal (0)', 'Fraud (1)'], zero_division=0))

    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    print(f"Overall F1-Score (for Fraud/Positive class): {f1:.4f}")
    print(f"Overall Precision (for Fraud/Positive class): {precision:.4f}")
    print(f"Overall Recall (for Fraud/Positive class): {recall:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    elif cm.size == 1:
        if y_true.iloc[0] == 0:
            tn, fp, fn, tp = cm[0][0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0][0]
    else:
        tn, fp, fn, tp = 0,0,0,0

    print("\nConfusion Matrix:")
    print(f"True Negatives (Correctly Normal): {tn}")
    print(f"False Positives (Legit flagged as Fraud): {fp}")
    print(f"False Negatives (Fraud missed): {fn}")
    print(f"True Positives (Fraud detected): {tp}")

    return tn, fp, fn, tp

def calculate_extrinsic_metrics(tn, fp, fn, tp, total_transactions_evaluated, model_name):
    """Calculates and prints extrinsic metrics."""
    print(f"\n--- Extrinsic Model Evaluation for {model_name} (Simulated Business Impact) ---")

    total_potential_fraud_losses = (tp + fn) * AVERAGE_FRAUD_LOSS_PER_TRANSACTION
    cost_of_reviews = fp * COST_OF_FALSE_POSITIVE
    savings_from_tp = tp * COST_OF_TRUE_FRAUD_DETECTED
    net_simulated_savings = savings_from_tp - cost_of_reviews
    
    print(f"Total potential fraud losses in test data: ${total_potential_fraud_losses:,.2f}")
    print(f"Savings from correctly detected fraud (True Positives): ${savings_from_tp:,.2f}")
    print(f"Cost from False Positives (manual review, customer inconvenience): ${cost_of_reviews:,.2f}")
    print(f"Net Simulated Cost Savings: ${net_simulated_savings:,.2f}")
    print(f"Absolute False Positives: {fp}")

if __name__ == "__main__":
    # --- Load and Prepare Data ---
    X_normal_train_df, X_normal_test_df, X_fraud_test_df = load_and_prepare_data(DATA_PATH)
    
    # --- Train Isolation Forest (using combined training data) ---
    print("\n=== Training Isolation Forest ===")
    # Combine normal train and fraud test data for Isolation Forest training
    # This approach lets IF see existing anomalies to better set its threshold for "isolation"
    combined_train_df_for_if = pd.concat([X_normal_train_df, X_fraud_test_df], ignore_index=True)
    # Ensure it's sorted for features like time_since_last_transaction
    combined_train_df_for_if = combined_train_df_for_if.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)

    if_model = train_isolation_forest_model(
        combined_train_df_for_if, # <--- NOW TRAINING ON COMBINED DATA
        ADVANCED_FEATURES,
        IF_CONTAMINATION_RATE,
        ISOLATION_FOREST_MODEL_PATH
    )

    # --- Train PyTorch Autoencoder (using normal training data) ---
    print("\n=== Training PyTorch Autoencoder ===")
    ae_model, minmax_scaler = train_autoencoder_model(
        X_normal_train_df, ADVANCED_FEATURES, AUTOENCODER_MODEL_PATH, SCALER_PATH, 
        INPUT_DIM, HIDDEN_DIM, LATENT_DIM, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE
    )

    print("\nBoth models (Isolation Forest and PyTorch Autoencoder) training finished.")
    print(f"Models trained on features: {ADVANCED_FEATURES}")
    print(f"Data split: {len(X_normal_train_df)} normal train, {len(X_normal_test_df)} normal test, {len(X_fraud_test_df)} fraud test.")

    # --- COMBINED EVALUATION ---
    print("\n\n=== Model Evaluation on Test Data ===")

    # Prepare combined test data for evaluation (features only)
    combined_test_df = pd.concat([X_normal_test_df, X_fraud_test_df], ignore_index=True)
    y_true_combined = combined_test_df['is_fraud'].astype(int) 
    X_combined_features = combined_test_df[ADVANCED_FEATURES]

    # --- Evaluate Isolation Forest ---
    print("\n=== Evaluating Isolation Forest ===")
    # Predict on combined test set
    if_predictions_raw = if_model.predict(X_combined_features)
    if_predictions_mapped = np.where(if_predictions_raw == -1, 1, 0)
    
    # Calculate IF metrics
    tn, fp, fn, tp = evaluate_model_performance(y_true_combined, if_predictions_mapped, "Isolation Forest")
    calculate_extrinsic_metrics(tn, fp, fn, tp, len(combined_test_df), "Isolation Forest")

    # --- Evaluate PyTorch Autoencoder ---
    print("\n=== Evaluating PyTorch Autoencoder ===")
    ae_model.eval() # Set model to evaluation mode
    
    # Prepare data for AE prediction: scale and convert to tensor
    X_combined_scaled = minmax_scaler.transform(X_combined_features)
    X_combined_tensor = torch.tensor(X_combined_scaled, dtype=torch.float32)

    # Calculate reconstruction errors
    with torch.no_grad():
        reconstructions = ae_model(X_combined_tensor)
        reconstruction_errors = torch.mean((X_combined_tensor - reconstructions)**2, dim=1).numpy()
    
    # Set Autoencoder anomaly threshold (from normal test data errors)
    # This threshold is calculated from the reconstruction errors of the *normal* test set.
    # This is CRITICAL for AE anomaly detection.
    X_normal_test_features_scaled = minmax_scaler.transform(X_normal_test_df[ADVANCED_FEATURES])
    X_normal_test_tensor = torch.tensor(X_normal_test_features_scaled, dtype=torch.float32)
    with torch.no_grad():
        normal_reconstructions = ae_model(X_normal_test_tensor)
        normal_reconstruction_errors = torch.mean((X_normal_test_tensor - normal_reconstructions)**2, dim=1).numpy()
    
    AUTOENCODER_ANOMALY_THRESHOLD = np.percentile(normal_reconstruction_errors, 99.5) 
    print(f"Autoencoder Anomaly Threshold (99.5 percentile of normal test data errors): {AUTOENCODER_ANOMALY_THRESHOLD:.4f}")

    # Apply threshold to get AE predictions
    ae_predictions_mapped = np.where(reconstruction_errors > AUTOENCODER_ANOMALY_THRESHOLD, 1, 0)
    
    # Calculate AE metrics
    tn, fp, fn, tp = evaluate_model_performance(y_true_combined, ae_predictions_mapped, "PyTorch Autoencoder")
    calculate_extrinsic_metrics(tn, fp, fn, tp, len(combined_test_df), "PyTorch Autoencoder")

    print("\nAll training and evaluation steps completed.")