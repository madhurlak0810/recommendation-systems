from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import pandas as pd
import io
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os

load_dotenv()
# Load AWS credentials from environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError("AWS credentials are not set in environment variables.")

app = FastAPI()

# AWS S3 Configuration
AWS_REGION = "us-east-1"
S3_BUCKET = "your-recommender-data"
S3_FILE_KEY = "reviews/user_ratings.csv"

# Initialize S3 Client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# Define the expected review data format
class Review(BaseModel):
    user: str
    name: str
    rating: float
    main_category: str
    sub_category: str

@app.post("/submit-review/")
async def submit_review(review: Review):
    try:
        # Try to fetch existing CSV from S3
        try:
            csv_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_FILE_KEY)
            existing_data = pd.read_csv(csv_obj['Body'],index_col=0)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                # If file doesn't exist, start a new DataFrame
                print("file not found, resolving to empty DataFrame")
                
                existing_data = pd.DataFrame(columns=review.dict().keys())
            else:
                raise e

        # Convert incoming review to DataFrame
        new_row = pd.DataFrame([review.dict()])

        # Append new row
        updated_data = pd.concat([existing_data, new_row], ignore_index=False)
        
        updated_data.reset_index(drop=True, inplace=True)

        # Convert DataFrame to CSV in memory
        csv_buffer = io.StringIO()
        updated_data.to_csv(csv_buffer, index=True)

        # Upload back to S3
        s3_client.put_object(Bucket=S3_BUCKET, Key=S3_FILE_KEY, Body=csv_buffer.getvalue())

        return {"message": "Review submitted successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""
How to run the FastAPI app and test the /submit-review/ endpoint:

1. **Start the FastAPI App**  
Make sure your virtual environment is activated and run:

    python -m uvicorn input.app.main:app --reload

This will start the server at: http://127.0.0.1:8000

---

2. **Test the /submit-review/ Endpoint Using cURL**

Use the following command to submit a sample review:

    curl -X POST "http://127.0.0.1:8000/submit-review/" \
    -H "Content-Type: application/json" \
    -d '{"user": "userTest", "name": "p1", "rating": 5, "main_category": "test", "sub_category": "sub"}'

If successful, you should get a response like:

    {"message":"Review submitted successfully."}

This will append the review data as a new row in the CSV stored in your configured S3 bucket.

---
3. **Note:**
- Ensure your `.env` file is properly configured with AWS credentials and S3 bucket details before running.
- The server must be running when you execute the `curl` command.
"""
