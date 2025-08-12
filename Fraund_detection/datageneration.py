import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
from faker import Faker
import random
import json
import boto3 # <--- Commented out for CSV generation focus
import base64 # <--- Commented out for CSV generation focus

# Initialize Faker for realistic data
fake = Faker()

def generate_user_profiles(num_users):
    """Generates a dictionary of user profiles with typical spending habits and locations."""
    user_profiles = {}
    for i in range(num_users):
        user_id = f"user_{str(uuid.uuid4())[:8]}" # Shorter user ID for readability
        user_profiles[user_id] = {
            "name": fake.name(),
            "avg_transaction_amount": round(random.uniform(20, 150), 2),
            "std_transaction_amount": round(random.uniform(5, 30), 2),
            "avg_transactions_per_day": random.randint(1, 5),
            "home_city": fake.city(),
            "home_country": fake.country(),
            "base_lat": float(fake.latitude()),
            "base_lon": float(fake.longitude()),
            "payment_methods": random.sample(["Credit Card", "Debit Card", "PayPal", "Bank Transfer"], k=random.randint(1, 3))
        }
    return user_profiles

def generate_transaction(
    user_id,
    user_profile,
    base_timestamp,
    is_fraudulent=False,
    previous_transaction_time=None
):
    """Generates a single transaction, with potential anomalies if is_fraudulent is True."""

    timestamp = base_timestamp + timedelta(seconds=random.randint(1, 300)) # Random time within a window

    # Default legitimate transaction
    amount = max(1.0, round(np.random.normal(user_profile["avg_transaction_amount"], user_profile["std_transaction_amount"]), 2))
    merchant = fake.company()
    payment_method = random.choice(user_profile["payment_methods"])
    # Slightly vary location around home city
    latitude = user_profile["base_lat"] + random.uniform(-0.1, 0.1)
    longitude = user_profile["base_lon"] + random.uniform(-0.1, 0.1)
    country = user_profile["home_country"] # Mostly transactions in home country
    city = user_profile["home_city"] # Mostly transactions in home city

    # --- Introduce Fraudulent Characteristics ---
    if is_fraudulent:
        fraud_type = random.choice(["high_amount", "foreign_location", "velocity_attack", "stolen_card"])

        if fraud_type == "high_amount":
            amount = round(np.random.uniform(user_profile["avg_transaction_amount"] * 5, user_profile["avg_transaction_amount"] * 20), 2)
            merchant = fake.domain_word().capitalize() + " Store" # Less common merchants
            payment_method = random.choice(["Credit Card", "Debit Card"]) # Often specific types for fraud
            city = fake.city() # Still domestic but potentially new city
            country = user_profile["home_country"] # Still domestic but potentially new city

        elif fraud_type == "foreign_location":
            amount = max(1.0, round(np.random.normal(user_profile["avg_transaction_amount"], user_profile["std_transaction_amount"] * 1.5), 2))
            latitude = float(fake.latitude()) # Completely new random location
            longitude = float(fake.longitude())
            country = fake.country() # Definitely a new country
            while country == user_profile["home_country"]: # Ensure it's different
                country = fake.country()
            city = fake.city()
            merchant = fake.domain_word().capitalize() + " Global"
            payment_method = random.choice(["Credit Card", "Debit Card"]) # Often specific types for fraud

        elif fraud_type == "velocity_attack":
            # Small amounts, very rapid succession. This type needs to be handled by the generator calling this function
            amount = round(np.random.uniform(5, 50), 2) # Smaller amounts
            merchant = fake.bs() # Obscure merchant name
            payment_method = random.choice(["Credit Card"]) # Often just one method
            # Location is typically same as last legit one or slightly off
            latitude = user_profile["base_lat"] + random.uniform(-0.05, 0.05)
            longitude = user_profile["base_lon"] + random.uniform(-0.05, 0.05)
            country = user_profile["home_country"]
            city = user_profile["home_city"]

            # If this is part of a velocity attack, timestamp should be very close to previous
            if previous_transaction_time:
                timestamp = previous_transaction_time + timedelta(seconds=random.randint(1, 10)) # Very quick succession
            else: # If it's the first in a velocity attack, make it close to last known legit tx
                timestamp = base_timestamp + timedelta(seconds=random.randint(1, 60))


        elif fraud_type == "stolen_card":
            amount = round(np.random.uniform(user_profile["avg_transaction_amount"] * 2, user_profile["avg_transaction_amount"] * 10), 2)
            merchant = fake.company() + " Online"
            # Simulate a new, "stolen" card number that is not in the user's usual profile
            payment_method = f"Credit Card {fake.credit_card_number()[-4:]} (STOLEN)"
            latitude = float(fake.latitude())
            longitude = float(fake.longitude())
            country = fake.country()
            city = fake.city()

    transaction_data = {
        "transaction_id": str(uuid.uuid4()),
        "timestamp": timestamp.isoformat(), # Use ISO format for easy parsing later
        "user_id": user_id,
        "merchant": merchant,
        "amount": amount,
        "currency": "USD", # For simplicity, stick to USD
        "payment_method": payment_method,
        "latitude": latitude,
        "longitude": longitude,
        "city": city,
        "country": country,
        "home_lat": user_profile["base_lat"],  # <--- ADDED THIS LINE
        "home_lon": user_profile["base_lon"],  # <--- ADDED THIS LINE
        "is_fraud": is_fraudulent # Ground truth label
    }
    return transaction_data

# --- Existing generate_simulated_payment_data function (for CSV output) ---
def generate_simulated_payment_data(
    num_users=100,
    total_transactions=10000,
    fraud_rate=0.005, # 0.5% fraud
    start_date="2023-01-01 00:00:00"
):
    """
    Generates a DataFrame of simulated payment transactions with a configurable fraud rate.
    """
    print(f"Generating {total_transactions} transactions for {num_users} users...")

    user_profiles = generate_user_profiles(num_users)
    user_ids = list(user_profiles.keys())
    
    transactions = []
    current_timestamp = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")

    last_user_transaction_time = {user_id: current_timestamp for user_id in user_ids}

    for i in range(total_transactions):
        if i % (total_transactions // 10) == 0 and total_transactions > 100:
            print(f"  {i}/{total_transactions} transactions generated...")

        chosen_user_id = random.choice(user_ids)
        chosen_user_profile = user_profiles[chosen_user_id]
        
        is_fraud = random.random() < fraud_rate

        if is_fraud and random.random() < 0.3:
             # Generate the first fraud transaction
            transactions.append(generate_transaction(
                chosen_user_id,
                chosen_user_profile,
                current_timestamp,
                is_fraudulent=True,
                previous_transaction_time=last_user_transaction_time[chosen_user_id]
            ))
            last_user_transaction_time[chosen_user_id] = datetime.fromisoformat(transactions[-1]["timestamp"])

            # Generate 1-3 more very rapid small frauds for this user
            num_velocity_tx = random.randint(1, 3)
            for _ in range(num_velocity_tx):
                 transactions.append(generate_transaction(
                    chosen_user_id,
                    chosen_user_profile,
                    current_timestamp,
                    is_fraudulent=True,
                    previous_transaction_time=last_user_transaction_time[chosen_user_id]
                ))
                 last_user_transaction_time[chosen_user_id] = datetime.fromisoformat(transactions[-1]["timestamp"])

        else:
            transactions.append(generate_transaction(
                chosen_user_id,
                chosen_user_profile,
                current_timestamp,
                is_fraudulent=is_fraud,
                previous_transaction_time=last_user_transaction_time[chosen_user_id]
            ))
            last_user_transaction_time[chosen_user_id] = datetime.fromisoformat(transactions[-1]["timestamp"])

        current_timestamp += timedelta(seconds=random.randint(1, 10))

    df = pd.DataFrame(transactions)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    print("\nData generation complete.")
    print(f"Total transactions generated: {len(df)}")
    print(f"Number of fraudulent transactions: {df['is_fraud'].sum()}")
    print(f"Fraud rate: {df['is_fraud'].sum() / len(df):.4f}")
    return df

# --- NEW generate_and_invoke_lambda_transactions function (for direct Lambda invocation) ---
def generate_and_invoke_lambda_transactions(
    num_users=100,
    total_transactions=1000,
    fraud_rate=0.005,
    start_date="2023-01-01 00:00:00",
    lambda_function_name='PaymentAnomalyDetectorML_BuildX',
    lambda_region='us-east-1'
):
    """
    Generates simulated payment transactions and directly invokes an AWS Lambda function for each.
    """
    print(f"Generating {total_transactions} transactions for {num_users} users and invoking Lambda: {lambda_function_name}...")

    user_profiles = generate_user_profiles(num_users)
    user_ids = list(user_profiles.keys())
    
    # Initialize Lambda client
    lambda_client = boto3.client('lambda', region_name=lambda_region)

    current_timestamp = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    last_user_transaction_time = {user_id: current_timestamp for user_id in user_ids}

    fraud_count = 0
    invoked_count = 0

    for i in range(total_transactions):
        if i % (total_transactions // 10) == 0 and total_transactions > 100:
            print(f"  {i}/{total_transactions} transactions generated...")

        chosen_user_id = random.choice(user_ids)
        chosen_user_profile = user_profiles[chosen_user_id]
        
        is_fraud = random.random() < fraud_rate

        transactions_to_invoke = []

        if is_fraud and random.random() < 0.3:
            # Velocity attack
            tx = generate_transaction(chosen_user_id, chosen_user_profile, current_timestamp, is_fraudulent=True, previous_transaction_time=last_user_transaction_time[chosen_user_id])
            transactions_to_invoke.append(tx)
            last_user_transaction_time[chosen_user_id] = datetime.fromisoformat(tx["timestamp"])
            fraud_count += 1

            num_velocity_tx = random.randint(1, 3)
            for _ in range(num_velocity_tx):
                tx_velocity = generate_transaction(chosen_user_id, chosen_user_profile, current_timestamp, is_fraudulent=True, previous_transaction_time=last_user_transaction_time[chosen_user_id])
                transactions_to_invoke.append(tx_velocity)
                last_user_transaction_time[chosen_user_id] = datetime.fromisoformat(tx_velocity["timestamp"])
                fraud_count += 1
        else:
            tx = generate_transaction(chosen_user_id, chosen_user_profile, current_timestamp, is_fraudulent=is_fraud, previous_transaction_time=last_user_transaction_time[chosen_user_id])
            transactions_to_invoke.append(tx)
            if is_fraud:
                fraud_count += 1
        
        for tx_data in transactions_to_invoke:
            # Lambda expects a Kafka-like event structure
            kafka_event_format = {
              "eventSource": "custom:direct-invoke",
              "records": {
                "payments-0": [
                  {
                    "topic": "payments-direct-stream",
                    "partition": 0,
                    "offset": invoked_count,
                    "timestamp": int(datetime.now().timestamp() * 1000),
                    "timestampType": "CREATE_TIME",
                    "key": None,
                    "value": base64.b64encode(json.dumps(tx_data).encode('utf-8')).decode('utf-8')
                  }
                ]
              }
            }
            
            try:
                response = lambda_client.invoke(
                    FunctionName=lambda_function_name,
                    InvocationType='Event',
                    Payload=json.dumps(kafka_event_format)
                )
                invoked_count += 1
            except Exception as e:
                print(f"Error invoking Lambda for transaction {tx_data.get('transaction_id')}: {e}")

        current_timestamp += timedelta(seconds=random.randint(1, 10))
        
    print("\nData invocation complete.")
    print(f"Total transactions generated: {total_transactions}")
    print(f"Total Lambda invocations: {invoked_count}")
    print(f"Number of fraudulent transactions generated: {fraud_count}")
    print(f"Simulated fraud rate: {fraud_count / total_transactions:.4f}")


if __name__ == "__main__":
    # --- Option 1: Generate CSV for Model Training (Current Focus) ---
    """ print("Generating data to CSV for model training...")
    simulated_df = generate_simulated_payment_data(
        num_users=100, # Use 100 users for this run
        total_transactions=10000, # Generate 10,000 transactions as requested
        fraud_rate=0.005,
        start_date="2023-01-01 00:00:00"
    )

    output_filename = "simulated_transactions.csv"
    simulated_df.to_csv(output_filename, index=False)
    print(f"\nSimulated data saved to {output_filename}") """

    # --- Option 2: Direct Lambda Invocation (Commented Out for now) ---
    print("\n(Lambda invocation is currently commented out in __main__.)")
    generate_and_invoke_lambda_transactions(
        num_users=100,
        total_transactions=1000,
        fraud_rate=0.05,
        start_date="2023-01-01 00:00:00",
        lambda_function_name='PaymentAnomalyDetectorML',
        lambda_region='us-east-1'
     )
    print("\nProducer script finished. Check AWS CloudWatch logs for Lambda invocation results.")