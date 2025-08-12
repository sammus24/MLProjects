import os
import boto3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get AWS credentials and region from environment variables
aws_access_key_id = os.getenv("AKIASRUGBHSYT5GLKTN4")
aws_secret_access_key = os.getenv("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")
aws_region = os.getenv("us-east-1")

print(f"Attempting to connect to AWS region: {aws_region}")

try:
    # Initialize the Bedrock client
    bedrock_client = boto3.client(
        service_name="bedrock",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    # List available foundation models to test access
    print("Listing available Amazon Bedrock Foundation Models...")
    response = bedrock_client.list_foundation_models(
        byOutputModality='TEXT', # Only list text-generating models for simplicity
        #byProvider='Anthropic' # Optional: Filter by provider if you know which one you'll use
    )

    models = response.get("modelSummaries", [])
    if models:
        print(f"Successfully connected to Bedrock. Found {len(models)} text-generating models.")
        print("First 5 models found:")
        for model in models[:5]:
            print(f"- {model['modelName']} (ID: {model['modelId']})")
    else:
        print("Successfully connected to Bedrock, but no text-generating models found in this region or with applied filters.")

except Exception as e:
    print(f"Error connecting to Bedrock or listing models: {e}")
    print("Please ensure your AWS credentials are correct, Bedrock is available in your region, and your IAM user/role has sufficient permissions (e.g., AmazonBedrockFullAccess).")

print("\nTest complete.")