import boto3
import requests
from io import BytesIO, StringIO
import pandas as pd

global colab_count
colab_count = 0

def query_colab_model(hostname, user):
    global colab_count

    # retrieve user_ratings
    bucket_name = "your-recommender-data"
    prefix = "reviews/user_ratings.csv"
    s3 = boto3.client("s3")
    s3_obj = s3.get_object(Bucket=bucket_name, Key=prefix)
    df = pd.read_csv(BytesIO(s3_obj["Body"].read()))

    # Define the base URL for your FastAPI app
    url = f"http://{hostname}/get_products"

    # Define the parameters (query string)
    params = {
        "top_n": 5,  # Get the top 5 products
        "user": user
    }

    s3_obj = s3.get_object(Bucket=bucket_name, Key=prefix)
    df = pd.read_csv(BytesIO(s3_obj["Body"].read()))

    # Send the GET request to the API
    response = requests.get(url, params=params)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data = response.json()[params["user"]]
        print("Popular Products:", data)
    else:
        print(f"Error: {response.status_code} - {response.text}")

    for i, item in enumerate(data):
        # TODO: pick items that user has/hasn't bought and what ratings to give
        df.loc[len(df)+i] = [params["user"], item["name"], 5, item["main_category"], item["sub_category"]]

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Key=prefix, Body=csv_buffer.getvalue())

    colab_count += 1
    if colab_count%100 == 0:
        print("retraining model")
        url = f"http://{hostname}/train_model"
        response = requests.get(url)
    print("done")


if __name__ == "__main__":
    query_colab_model("0.0.0.0:8000", "user0")