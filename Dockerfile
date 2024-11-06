# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Copy the rest of the application
COPY . /app

# Set working directory
WORKDIR /app


# Install system dependencies required for Prophet
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt



# Expose the port the app runs on
EXPOSE 3030

# Command to run the application
CMD ["python", "visualisation/dash_visualisation_prophet.py"] 