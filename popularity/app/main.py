from fastapi import FastAPI
from apps.routes import router, df

import boto3
import pandas as pd
import io

app = FastAPI()
app.include_router(router)

bucket_name = "your-recommender-data"
prefix      = "cleaned/models.csv"

@app.on_event("startup")
def load_data_from_s3():
    global df
    s3    = boto3.client("s3")
    obj   = s3.get_object(Bucket=bucket_name, Key=prefix)
    df_s3 = pd.read_csv(io.BytesIO(obj["Body"].read()))

    df.clear()
    df.update(df_s3)

