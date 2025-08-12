# Use the official AWS Lambda Python 3.11 base image
# This aligns with the requirements of the chosen numpy/pandas/sklearn versions.
FROM public.ecr.aws/lambda/python:3.11 

# Set working directory in the container
WORKDIR /var/task

# Install build tools needed for *some* packages or if wheels aren't found.
# Keeping this line for robustness, as some scientific packages can still try to compile.
# '--assumeyes' (or -y) automatically answers yes to prompts.
RUN yum install -y gcc gcc-c++ lapack-devel blas-devel python3-devel

# Copy your Lambda function code, requirements, and model
COPY requirements.txt .
COPY lambda_function.py .
COPY isolation_forest_model.joblib .

# Install Python dependencies from requirements.txt
# --no-cache-dir reduces image size slightly
RUN pip install -r requirements.txt --target . --no-cache-dir

# Set the CMD to your handler (function_name.handler_name)
CMD [ "lambda_function.lambda_handler" ]