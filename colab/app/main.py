from fastapi import FastAPI, Query
from contextlib import asynccontextmanager
from typing import Optional
import pandas as pd
import numpy as np
import joblib
from app.routes import router
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
from clearml import Task, Logger

def make_collab_model(df: pd.DataFrame):
    # clearml task setup
    task: Task = Task.init(
        project_name="recommendation-systems",
        task_name="colab model training",
    )
    logger = Logger.current_logger()

    # record parameters
    params = {
        "no_components": 10,
        "loss": "warp",
        "num_epochs": 30,
        "k": 5,
    }
    task.connect(params)

    # prepare dataset
    dataset = Dataset()
    dataset.fit(df['user'], df['name'])

    (interactions, weights) = dataset.build_interactions([(row['user'], row['name'], row['rating']) for _, row in df.iterrows()])

    # train model
    model = LightFM(no_components=params["no_components"], loss=params["loss"])
    for epoch in range(params["num_epochs"]):
        # only fit partial
        model.fit_partial(interactions, epochs=1, num_threads=4)

        precision = precision_at_k(model, interactions, k=params["k"]).mean()

        logger.report_scalar(f"Precision@{params['k']}", "Epoch", iteration=epoch, value=precision)
        print(f"Epoch {epoch}/{params['num_epochs']} - Precision@{params['k']}: {precision:.4f}")
    
    # save model artifacts
    joblib.dump(model, "lightfm_model.pkl")
    task.upload_artifact(name="lightfm_model", artifact_object="lightfm_model.pkl")
    joblib.dump(dataset, "dataset.pkl")
    task.upload_artifact("dataset", artifact_object="dataset.pkl")
    joblib.dump(interactions, "interactions.pkl")
    task.upload_artifact("interactions", artifact_object="interactions.pkl")
    task.close()

    return model, dataset, interactions

# Global storage
model = None
dataset = None
interactions = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, dataset, interactions

    try:
        # Initialize model at startup
        df = pd.read_csv("user_ratings.csv")
        model, dataset, interactions = make_collab_model(df)
        print("✅ Model initialized")
        yield
    finally:
        print("👋 Shutting down app... clean up here if needed")

app = FastAPI(lifespan=lifespan)
app.include_router(router)

# Allow access to df_clean from routes
def get_model():
    return model, dataset, interactions

# Inject dependency into router
router.model = get_model