from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import router, load_data_from_s3, set_global_df, clear_global_df

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load S3 data once
    df_s3 = load_data_from_s3()
    set_global_df(df_s3)
    print("Data loaded from S3 at startup")

    yield

    clear_global_df()
    print("Cleared data at shutdown")

app = FastAPI(lifespan=lifespan)

app.include_router(router)
