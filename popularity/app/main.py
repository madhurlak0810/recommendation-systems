from fastapi import FastAPI
from popularity.app.routes import load_data_from_s3, ProductNameSimilarityModel, download_joblib_from_s3
from popularity.app.routes import router
from contextlib import asynccontextmanager
# from app.routes import router
# from app.routes import load_data_from_s3, set_global_df, ProductNameSimilarityModel,download_joblib_from_s3
# from app.routes import routes
CSV_LOCAL_PATH = "data/models.csv"
MODEL_DIR      = "models/"

@asynccontextmanager
async def lifespan(app: FastAPI):
    df = load_data_from_s3()
    # Download the model from S3
    download_joblib_from_s3("your-recommender-data", "models/cosine_similarity_model.joblib", MODEL_DIR + "cosine_similarity_model.joblib")
    model = ProductNameSimilarityModel()
    model.load_model(MODEL_DIR + "cosine_similarity_model.joblib")
    # Initialize the model with the loaded data
    app.state.model = model
    app.state.df = df
    print("Startup complete: Data loaded and model initialized.")

    yield

    print("Shutdown complete.")

app = FastAPI\
(
    title="Product Name Similarity API",
    description="API for retrieving top products and similar product names",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)
