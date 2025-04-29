from fastapi import FastAPI
from app.routes import router, df

import boto3
import pandas as pd
import io
import os
from dotenv import load_dotenv

app = FastAPI()
app.include_router(router)

bucket_name = "your-recommender-data"
prefix      = "cleaned/models.csv"

load_dotenv()
@app.on_event("startup")
def load_data_from_s3():
    global df
    s3 = boto3.client\
    (
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = os.getenv("AWS_REGION", "us-east-1"),
    )
    obj   = s3.get_object(Bucket=bucket_name, Key=prefix)
    df_s3 = pd.read_csv(io.BytesIO(obj["Body"].read()))

    df.clear()
    df.update(df_s3)

