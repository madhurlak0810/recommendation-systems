from fastapi import FastAPI
from app.routes import router
from contextlib import asynccontextmanager
from app.routes import load_data_from_s3, set_global_df, ProductNameSimilarityModel

CSV_LOCAL_PATH = "data/models.csv"
MODEL_DIR      = "model/"

@asynccontextmanager
async def lifespan(app: FastAPI):
    df = load_data_from_s3()
    set_global_df(df)

    global model
    model = ProductNameSimilarityModel(csv_path=CSV_LOCAL_PATH, save_model_dir=MODEL_DIR)

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
