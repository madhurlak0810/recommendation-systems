#!/bin/bash

# Docker image name (replace with your own image name and tag)
IMAGE_NAME="jliu3714/update-app-container"
TAG="latest"  # You can replace 'latest' with a version or tag of your choice

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME:$TAG .

# Push the Docker image to Docker Hub
# echo "Pushing Docker image to Docker Hub..."
# docker push $IMAGE_NAME:$TAG

# Run the Docker container
echo "Running Docker container..."
docker run -p 8000:8000 -v ~/.aws:/root/.aws:ro \
    -e CLEARML_API_HOST=https://api.clear.ml -e CLEARML_API_ACCESS_KEY="FSI1HMV87LF8DPE09PQMZC18769MM9" -e CLEARML_API_SECRET_KEY="UZExi7iLWHzmk4s8qy0xDdMHvvz-V4kMkRP712YCf4c9veS6_SCCNW2j3uOnCCRxeps" \
    --name update-app-container $IMAGE_NAME:$TAG \

# Gracefully stop the container when the script is terminated
trap "docker stop update-app-container && docker rm update-app-container" EXIT