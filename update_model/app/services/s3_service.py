import boto3
import joblib
import os

def upload_joblib_to_s3(local_file_path, bucket_name, s3_key, aws_region='us-east-1'):
    """
    Uploads a local .joblib file to an S3 bucket.

    Parameters:
    - local_file_path: str, path to the local .joblib file
    - bucket_name: str, name of the S3 bucket
    - s3_key: str, destination key (i.e., file path) in the S3 bucket
    - aws_region: str, AWS region (default: 'us-east-1')
    """
    # Load joblib object to ensure file is readable (optional)
    try:
        model = joblib.load(local_file_path)
    except Exception as e:
        raise ValueError(f"Failed to load .joblib file: {e}")
    
    # Upload file to S3
    try:
        s3 = boto3.client('s3', region_name=aws_region)
        with open(local_file_path, 'rb') as f:
            s3.upload_fileobj(f, bucket_name, s3_key)
        print(f"Successfully uploaded {local_file_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload to S3: {e}")
