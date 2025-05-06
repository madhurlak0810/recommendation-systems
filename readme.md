# Project Overview

This repository contains multiple components for building, deploying, and managing machine learning models and APIs. The project is organized into several subdirectories, each serving a specific purpose, such as collaborative filtering, hybrid models, input processing, popularity-based recommendations, and model updates.

## Folder Structure

### Root Directory
- **`.gitignore`**: Specifies files and directories to be ignored by Git.
- **`data.ipynb`**: A Jupyter notebook for data exploration or preprocessing.
- **`model_query.py`**: A Python script for querying models.
- **`readme.md`**: This file, providing an overview of the project.
- **`Rohan_v2.py`**: A Python script, likely related to a specific functionality or experiment.
- **`s3_data.py`**: A script for interacting with AWS S3.
- **`setup.py`**: A setup script for packaging the project.

---

### `colab/`
Contains files related to collaborative filtering models.
- **`api_query_example.py`**: Example script for querying the API.
- **`app/`**: Contains the FastAPI application code for collaborative filtering.
- **`build.sh`**: A script to build and run the Docker container for the collaborative filtering app.
- **`collab_model.ipynb`**: Jupyter notebook for training or analyzing the collaborative filtering model.
- **`collab_model_retrained.ipynb`**: Jupyter notebook for retraining the collaborative filtering model.
- **`Dockerfile`**: Dockerfile for building the collaborative filtering app container.
- **`fake.py`**: A placeholder or utility script.
- **`k8s/`**: Kubernetes configuration files for deploying the collaborative filtering app.
- **`requirements.txt`**: Python dependencies for the collaborative filtering app.

---

### `hybrid/`
Contains files for hybrid recommendation models.
- **`app/`**: Contains the FastAPI application code for hybrid models.
- **`Dockerfile`**: Dockerfile for building the hybrid model app container.
- **`requirements.txt`**: Python dependencies for the hybrid model app.

---

### `input/`
Contains files for input processing.
- **`app/`**: Contains the FastAPI application code for input processing.
- **`Dockerfile`**: Dockerfile for building the input processing app container.
- **`requirements.txt`**: Python dependencies for the input processing app.

---

### `popularity/`
Contains files for popularity-based recommendation models.
- **`app/`**: Contains the FastAPI application code for popularity-based recommendations.
- **`Dockerfile`**: Dockerfile for building the popularity-based app container.
- **`requirements.txt`**: Python dependencies for the popularity-based app.

---

### `update_model/`
Contains files for updating models.
- **`app/`**: Contains the application code for model updates.
- **`build.sh`**: A script to build and run the Docker container for the model update app.
- **`Dockerfile`**: Dockerfile for building the model update app container.

---

### `k8s/`
Contains Kubernetes deployment configurations.
- **`colab-deployment.yaml`**: Deployment configuration for the collaborative filtering app.
- **`hybrid-deployment.yaml`**: Deployment configuration for the hybrid model app.
- **`input-deployment.yaml`**: Deployment configuration for the input processing app.
- **`popularity-deployment.yaml`**: Deployment configuration for the popularity-based app.
- **`update-deployment.yaml`**: Deployment configuration for the model update app.

---

## How to Use

### Building and Running Docker Containers
Each subdirectory contains a `Dockerfile` and a `build.sh` script (where applicable) to build and run the Docker containers. For example:
1. Navigate to the desired subdirectory (e.g., `colab/`).
2. Run the `build.sh` script:
   ```sh
   ./build.sh


Kubernetes Deployment
Kubernetes deployment files are located in the k8s directory. Use kubectl to apply the configurations:

Dependencies
Each subdirectory contains a requirements.txt file specifying the Python dependencies for that component. Install them using:

Notes
The project uses FastAPI for building APIs.
Docker is used for containerization, and Kubernetes is used for orchestration.
ClearML is integrated for experiment tracking and model management.
AWS S3 is used for data storage.
