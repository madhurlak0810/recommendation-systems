#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t colab-app .

# Run the Docker container
echo "Running Docker container..."
docker run -p 8000:8000 colab-app