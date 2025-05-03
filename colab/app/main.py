from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import router, model_path, download_joblib_from_s3
import joblib

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Initialize model at startup and store in app.state
        download_joblib_from_s3("your-recommender-data", "models/lightfm_collab_model.joblib", model_path)
        model_data = joblib.load(model_path)
        app.state.model = model_data['model']
        app.state.dataset = model_data['dataset']
        app.state.interactions = model_data['interactions']
        print("✅ Model initialized")
        yield
    finally:
        print("👋 Shutting down app... clean up here if needed")

app = FastAPI(lifespan=lifespan)
app.include_router(router)