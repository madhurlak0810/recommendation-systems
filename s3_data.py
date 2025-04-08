import boto3
import os
import pandas as pd
import numpy as np
import sys
import tqdm
import glob

def download_data(s3, local_dir):
    for obj in tqdm.tqdm(response["Contents"]):
        s3.download_file(bucket_name, obj["Key"], os.path.join(local_dir, os.path.basename(obj["Key"])))

if __name__ == "__main__":
    # usage: python s3_data.py user_accessKeys.csv 
    bucket_name = "your-recommender-data"
    prefix = "archive (1)/"
    local_dir = "data/"
    os.makedirs(local_dir, exist_ok=True)

    # connect to s3 instance
    if len(sys.argv) > 1:
        df_cred = pd.read_csv(sys.argv[1])
        s3 = boto3.client(
            "s3", 
            aws_access_key_id=df_cred["Access key ID"].values[0], 
            aws_secret_access_key=df_cred["Secret access key"].values[0], 
            region_name="us-east-1")
    else:
        s3 = boto3.client("s3")

    # get data from s3 instance
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    download_data(s3, local_dir)