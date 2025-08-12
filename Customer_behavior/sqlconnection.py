import mysql.connector

# MySQL connection details - FILL THESE IN!
DB_CONFIG = {
    'host': 'localhost', # Or your MySQL host IP/domain
    'user': 'root',
    'password': 'root',
    'port': 3306 # Default MySQL port
}

DB_NAME = 'ecommerce_customer_behavior'

try:
    # Connect to MySQL server (without specifying a database first)
    cnxn = mysql.connector.connect(**DB_CONFIG)
    cursor = cnxn.cursor()

    # Create the database if it doesn't exist
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    print(f"Database '{DB_NAME}' ensured to exist.")

    # Close initial connection
    cursor.close()
    cnxn.close()

except mysql.connector.Error as err:
    print(f"Error connecting to MySQL or creating database: {err}")
    # Handle specific errors if needed
    if err.errno == mysql.connector.errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
    elif err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
    else:
        print(err)
    exit() # Exit if we can't connect or create DB