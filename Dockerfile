FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv energythuasapp
ENV PATH="/app/energythuasapp/bin:$PATH"
SHELL ["/bin/bash", "-c"]
RUN source energythuasapp/bin/activate

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install requirements
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8030

# Command to run the application with the virtual environment
CMD ["energythuasapp/bin/python", "visualisation/dash_visualisation_prophet.py"] 