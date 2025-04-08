import boto3
import pandas as pd
from io import BytesIO

class S3Service:
    def __init__(self, credentials_path=None, region="us-east-1"):
        if credentials_path:
            creds = pd.read_csv(credentials_path)
            self.s3 = boto3.client(
                "s3",
                aws_access_key_id=creds["Access key ID"].values[0],
                aws_secret_access_key=creds["Secret access key"].values[0],
                region_name=region
            )
        else:
            self.s3 = boto3.client("s3")

    def list_csv_keys(self, bucket_name, prefix):
        response = self.s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        return [
            obj["Key"]
            for obj in response.get("Contents", [])
            if obj["Key"].endswith(".csv")
        ]

    def read_csv_from_s3(self, bucket_name, key):
        obj = self.s3.get_object(Bucket=bucket_name, Key=key)
        return pd.read_csv(BytesIO(obj["Body"].read()))

    def upload_csv_to_s3(self, df, bucket_name, key):
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        self.s3.put_object(Bucket=bucket_name, Key=key, Body=csv_buffer.getvalue())
