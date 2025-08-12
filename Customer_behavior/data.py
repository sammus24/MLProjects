import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker for realistic data
fake = Faker('en_US')

# --- Configuration Parameters ---
NUM_CUSTOMERS = 5000 # Number of unique customers
AVG_TRANSACTIONS_PER_CUSTOMER = 15 # Average transactions per customer
TRANSACTION_DATE_RANGE_YEARS = 3 # Data spanning 3 years
START_DATE = datetime(2022, 1, 1) # Start date for transactions

# --- Product and Channel Definitions ---
PRODUCT_CATEGORIES = [
    'Electronics', 'Clothing', 'Home Goods', 'Books', 'Food & Beverages',
    'Beauty', 'Sports & Outdoors', 'Toys', 'Automotive', 'Services'
]
PRODUCT_ITEMS = { # Simple mapping for variety
    'Electronics': ['Smartphone', 'Laptop', 'Headphones', 'Smartwatch', 'TV'],
    'Clothing': ['T-shirt', 'Jeans', 'Jacket', 'Dress', 'Sneakers'],
    'Home Goods': ['Coffee Maker', 'Blender', 'Vacuum Cleaner', 'Lamp', 'Sofa'],
    'Books': ['Fiction', 'Non-Fiction', 'Biography', 'Science Fiction'],
    'Food & Beverages': ['Coffee', 'Snacks', 'Juice', 'Gourmet Food'],
    'Beauty': ['Shampoo', 'Lotion', 'Makeup Kit', 'Perfume'],
    'Sports & Outdoors': ['Dumbbells', 'Yoga Mat', 'Tent', 'Running Shoes'],
    'Toys': ['Action Figure', 'Building Blocks', 'Board Game', 'Doll'],
    'Automotive': ['Car Cleaner', 'Tire Pump', 'Jump Starter'],
    'Services': ['Subscription', 'Consultation', 'Online Course']
}
CHANNELS = ['online', 'in_store', 'mobile_app', 'call_center']
PAYMENT_METHODS = ['Credit Card', 'PayPal', 'Debit Card', 'Bank Transfer', 'Cash']
SHIPPING_TYPES = ['Standard', 'Express', 'Next-Day', 'In-Store Pickup']

# Define churn rates (e.g., 10% overall churn)
CHURN_RATE = 0.10

# --- 1. Generate Customer Data ---
print("Generating customer data...")
customers_data = []
for i in range(NUM_CUSTOMERS):
    customer_id = f'CUST{100000 + i}'
    
    # Customer join date can be before or during the transaction period
    # Ensure it's not so far in the past it creates issues, nor too far in the future
    min_join_date = START_DATE - timedelta(days=365 * 2) # Up to 2 years before START_DATE
    max_join_date = START_DATE + timedelta(days=365 * (TRANSACTION_DATE_RANGE_YEARS - 1)) # Within the transaction range, giving time for activity
    customer_since = fake.date_between(start_date=min_join_date, end_date=max_join_date)
    
    customers_data.append({
        'customer_id': customer_id,
        'customer_name': fake.name(),
        'customer_age': random.randint(18, 75),
        'gender': random.choice(['Male', 'Female', 'Other']),
        'region': random.choice(['North', 'South', 'East', 'West', 'Central']),
        'customer_since': customer_since # This is a date object
    })
customer_df = pd.DataFrame(customers_data)
# Convert customer_since to datetime for easier comparison
customer_df['customer_since'] = pd.to_datetime(customer_df['customer_since'])

# --- ASSIGN CHURN STATUS TO THE DATAFRAME HERE ---
# This ensures 'churn' column exists for sure before iterating over customer_df for transactions
customer_df['churn'] = 0 # Default to retained

for index, row in customer_df.iterrows():
    churn_prob = CHURN_RATE
    
    # Adjust churn probability based on customer_since relative to current simulated 'now' (end of data period)
    # Customers who joined more recently might have a slightly higher initial churn rate
    # Very old, active customers might have lower churn
    end_of_data_period = START_DATE + timedelta(days=365 * TRANSACTION_DATE_RANGE_YEARS)
    days_customer_active_in_data = (end_of_data_period - row['customer_since']).days

    if days_customer_active_in_data < 180: # Less than 6 months active
        churn_prob *= 2.0 # Higher chance to churn
    elif days_customer_active_in_data > (365 * (TRANSACTION_DATE_RANGE_YEARS - 1)): # Active almost throughout the whole period
        churn_prob *= 0.5 # Lower chance to churn

    if random.random() < churn_prob:
        customer_df.at[index, 'churn'] = 1


# --- 2. Generate Transaction Data ---
print(f"Generating transaction data for {NUM_CUSTOMERS} customers...")
transactions_data = []
transaction_id_counter = 0

for _, customer_row in customer_df.iterrows(): 
    num_transactions = max(1, int(np.random.normal(AVG_TRANSACTIONS_PER_CUSTOMER, 5))) # Normal distribution for number of transactions
    
    # Determine the absolute latest possible transaction date for this customer
    # This is the end of the overall data period
    overall_end_date = START_DATE + timedelta(days=365 * TRANSACTION_DATE_RANGE_YEARS)
    
    # If the customer churned, their transactions should cease before the overall_end_date
    customer_active_until = overall_end_date # Default: active until end of data period
    
    if customer_row['churn'] == 1:
        # Churned customers' last transaction date must be:
        # 1. After their join date (plus some minimum activity period)
        # 2. Before the overall end date (with some buffer before actual churn date)
        
        min_last_tx_date = customer_row['customer_since'] + timedelta(days=60) # Must have at least 2 months of activity
        max_last_tx_date = overall_end_date - timedelta(days=30) # At least 1 month before end of data period
        
        # Ensure min_last_tx_date doesn't exceed max_last_tx_date (could happen if customer_since is very late)
        if min_last_tx_date >= max_last_tx_date:
            customer_active_until = max_last_tx_date # Force it to be at least this date if dates overlap
        else:
            customer_active_until = fake.date_time_between(start_date=min_last_tx_date, end_date=max_last_tx_date)

    # Ensure transaction start date is never before customer_since
    transaction_start_date_for_customer = customer_row['customer_since']
    
    transaction_dates = []
    # Generate transaction dates ensuring they are chronological and within active period
    current_tx_date = transaction_start_date_for_customer
    for _ in range(num_transactions):
        # Prevent "end_date < start_date" for faker
        if current_tx_date >= customer_active_until:
            tx_date = customer_active_until # If current date is past the end, set to end
        else:
            tx_date = fake.date_time_between(start_date=current_tx_date, end_date=customer_active_until)
        
        transaction_dates.append(tx_date)
        current_tx_date = tx_date # For next iteration, ensure it's after this one

    # Sort again just to be absolutely sure, especially if we didn't strictly increment current_tx_date
    transaction_dates.sort()

    for purchase_date in transaction_dates:
        transaction_id_counter += 1
        transaction_id = f'TRX{transaction_id_counter:07d}'
        
        category = random.choice(PRODUCT_CATEGORIES)
        item = random.choice(PRODUCT_ITEMS[category])
        price = round(random.uniform(5.0, 500.0), 2)
        quantity = random.randint(1, 5) if item != 'Subscription' else 1
        total_amount = round(price * quantity * (1 + random.uniform(-0.1, 0.1)), 2) # Add slight variability

        # Simulate channel preference - make some channels more common
        channel = random.choices(
            CHANNELS,
            weights=[0.5, 0.3, 0.15, 0.05], # online most common, then in-store, mobile, call_center
            k=1
        )[0]
        
        # Simulate shipping type based on channel
        shipping_type = 'In-Store Pickup' if channel == 'in_store' else random.choice([st for st in SHIPPING_TYPES if st != 'In-Store Pickup'])
        
        discount_applied = 1 if random.random() < 0.2 else 0 # 20% chance of discount
        promo_code_used = 1 if discount_applied == 1 and random.random() < 0.7 else 0 # Promo code if discount applied
        
        returns = 1 if random.random() < 0.08 else 0 # 8% chance of return
        
        review_rating = random.randint(1, 5) if random.random() < 0.7 else np.nan # 70% chance of rating, else NaN

        transactions_data.append({
            'transaction_id': transaction_id,
            'customer_id': customer_row['customer_id'],
            'purchase_date': purchase_date,
            'product_category': category,
            'product_item': item,
            'product_price': price,
            'quantity': quantity,
            'total_purchase_amount': total_amount,
            'channel': channel,
            'payment_method': random.choice(PAYMENT_METHODS),
            'shipping_type': shipping_type,
            'discount_applied': discount_applied,
            'promo_code_used': promo_code_used,
            'review_rating': review_rating,
            'returns': returns
        })

transaction_df = pd.DataFrame(transactions_data)
# Convert purchase_date to datetime for easier operations
transaction_df['purchase_date'] = pd.to_datetime(transaction_df['purchase_date'])

# --- 3. Merge Customer and Transaction Data ---
print("Merging customer and transaction data...")
# Merge customer_df's customer-level columns into transaction_df
full_df = pd.merge(transaction_df, customer_df, on='customer_id', how='left')

# --- 4. Final Touches & Export ---
print("Applying final touches and exporting...")

# Calculate "Frequency of Purchases" as the total count of transactions for each customer.
customer_frequency = full_df.groupby('customer_id')['transaction_id'].count().reset_index()
customer_frequency.rename(columns={'transaction_id': 'frequency_of_purchases'}, inplace=True)

# Determine preferred payment method by simple mode (most frequent)
# Using `dropna=False` ensures NaN rows are considered, and `first` breaks ties consistently
preferred_payment_method = full_df.groupby('customer_id')['payment_method'].apply(lambda x: x.mode(dropna=False)[0] if not x.mode().empty else np.nan).reset_index()
preferred_payment_method.rename(columns={'payment_method': 'preferred_payment_method'}, inplace=True)

# Merge these calculated features into the full_df
full_df = pd.merge(full_df, customer_frequency, on='customer_id', how='left')
full_df = pd.merge(full_df, preferred_payment_method, on='customer_id', how='left')

# Replace NaN ratings with 0 if you prefer, or handle during analysis
full_df['review_rating'].fillna(0, inplace=True)


# Reorder columns for clarity (optional, but good for consistent output)
final_columns = [
    'customer_id', 'customer_name', 'customer_age', 'gender', 'region', 'customer_since',
    'churn', # Customer-level churn
    'transaction_id', 'purchase_date', 'channel', 'product_category', 'product_item',
    'product_price', 'quantity', 'total_purchase_amount', 'payment_method',
    'shipping_type', 'discount_applied', 'promo_code_used', 'review_rating',
    'returns',
    'frequency_of_purchases', # Customer-level calculated frequency
    'preferred_payment_method' # Customer-level calculated preferred payment method
]
full_df = full_df[final_columns]

# Display some info
print("\nGenerated Data Info:")
full_df.info()
print("\nFirst 5 rows:")
print(full_df.head())
print("\nChannel distribution:")
print(full_df['channel'].value_counts())
print("\nChurn distribution (customer-level unique values):")
# Dropping duplicates to get customer-level churn count correctly
print(full_df[['customer_id', 'churn']].drop_duplicates()['churn'].value_counts())

# Save to CSV
CSV_FILE_PATH = 'simulated_ecommerce_data.csv'
full_df.to_csv(CSV_FILE_PATH, index=False)
print(f"\nDataset saved to {CSV_FILE_PATH}")

# Optional: Also save a 'customer_snapshot' for customer-level aggregated features
customer_snapshot_df = full_df.drop_duplicates(subset=['customer_id']).copy()
# Only keep the customer-level columns, and then add aggregated features
customer_snapshot_df = customer_snapshot_df[['customer_id', 'customer_name', 'customer_age', 'gender', 'region', 'customer_since', 'churn', 'frequency_of_purchases', 'preferred_payment_method']]
customer_snapshot_df['total_lifetime_spend'] = full_df.groupby('customer_id')['total_purchase_amount'].sum().reindex(customer_snapshot_df['customer_id']).values
customer_snapshot_df.to_csv('simulated_customer_snapshot.csv', index=False)
print(f"Customer snapshot saved to simulated_customer_snapshot.csv")