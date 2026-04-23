# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install kedro-datasets pyarrow datasets scikit-learn

# Copy the rest of the project
COPY . .

# Command to run the pipeline
CMD ["kedro", "run"]