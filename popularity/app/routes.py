from fastapi import APIRouter, Query
import pandas as pd
import boto3
import io
import os
from dotenv import load_dotenv

router = APIRouter()

df = pd.DataFrame()

def load_data_from_s3():
    load_dotenv()

    bucket_name = "your-recommender-data"
    prefix = "cleaned/models.csv"

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )

    obj = s3.get_object(Bucket=bucket_name, Key=prefix)
    df_s3 = pd.read_csv(io.BytesIO(obj["Body"].read()))
    return df_s3

def set_global_df(new_df):
    global df
    df.clear()
    df.update(new_df)

@router.get("/top_n_models")
def top_n_models(n: int = Query(10, ge=1, le=100)):
    top_n = df.sort_values(by="no_of_ratings", ascending=False).head(n)
    result = top_n[['name', 'no_of_ratings']].to_dict(orient="records")
    return {"top_n_models": result}
