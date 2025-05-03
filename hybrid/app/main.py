from fastapi import FastAPI
from app.routes import ProductHybridModel, download_joblib_from_s3
from app.routes import router
from contextlib import asynccontextmanager
import os

MODEL_DIR      = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Download the model from S3
    download_joblib_from_s3("your-recommender-data", "models/svd_model_user_data.joblib", MODEL_DIR + "svd_model_user_data.joblib")
    model = ProductHybridModel()
    model.load_model(MODEL_DIR + "svd_model_user_data.joblib")
    # Initialize the model with the loaded data
    app.state.model = model
    print("Startup complete: Data loaded and model initialized.")

    yield

    print("Shutdown complete.")

app = FastAPI\
(
    title="Hybrid Product Recommendation API",
    description="API for retrieving top products in correlation with user ratings",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)