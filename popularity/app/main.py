from fastapi import FastAPI, Query
from typing import Optional
import pandas as pd
import numpy as np
import boto3
import io
from app.routes import router
import asyncio

app = FastAPI()

df_clean =None

app.include_router(router)

# ---------- CONFIG ----------
BUCKET_NAME = "your-recommender-data"
S3_KEY = "cleaned/models.csv"
REGION = "us-east-1"  # change as needed
# ----------------------------

# Global storage
df_clean = None

def fetch_data_from_s3():
    s3 = boto3.client("s3", region_name=REGION)
    response = s3.get_object(Bucket=BUCKET_NAME, Key=S3_KEY)
    content = response["Body"].read()
    df = pd.read_csv(io.BytesIO(content))
    df = df.dropna(subset=["ratings", "no_of_ratings"])
    df["popularity_score"] = df["ratings"] * np.log1p(df["no_of_ratings"])
    return df

# Load data from S3 on startup
@app.on_event("startup")
async def startup_event():
    global df_clean

    async def refresh_loop():
        global df_clean
        while True:
            try:
                df_clean = fetch_data_from_s3()
                print("Data refreshed")
            except Exception as e:
                print(f"Failed to refresh data: {e}")
            await asyncio.sleep(300)  # refresh every 5 minutes

    df_clean = fetch_data_from_s3()
    # app.add_event_handler("startup", refresh_loop())
    asyncio.create_task(refresh_loop())


# Allow access to df_clean from routes
def get_data():
    return df_clean

# Inject dependency into router
router.df_loader = get_data