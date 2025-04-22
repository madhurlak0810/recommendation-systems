#!/bin/bash

# Docker image name (replace with your own image name and tag)
IMAGE_NAME="jliu3714/colab-app"
TAG="latest"  # You can replace 'latest' with a version or tag of your choice

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME:$TAG .

# Push the Docker image to Docker Hub
echo "Pushing Docker image to Docker Hub..."
docker push $IMAGE_NAME:$TAG

# Run the Docker container
echo "Running Docker container..."
docker run -p 8000:8000 --name colab-app-container $IMAGE_NAME:$TAG

# Gracefully stop the container when the script is terminated
trap "docker stop colab-app-container && docker rm colab-app-container" EXIT