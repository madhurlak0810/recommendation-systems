from fastapi import APIRouter, Query,Request
import pandas as pd
import boto3
import io
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter()

csv_local_path = "data/models.csv"

class ProductNameSimilarityModel:
    def __init__(self):
        self.product_names       = None
        self.vectorizer          = None
        self.tfidf_matrix        = None
        self.cosine_sim          = None
        self.product_indices     = None
        self.num_recommendations = None

        self.csv_path            = None
        self.save_model_dir      = None
        self.sample_size         = None
        self.model_path          = None
        self.num_recommendations = None



    def load_model(self, model_path):
        if self.cosine_sim is None or self.product_names is None or self.product_indices is None:
            try:
                model_data = joblib.load(model_path)
                self.product_names = model_data['product_names']
                self.vectorizer = model_data['vectorizer']
                self.tfidf_matrix = self.vectorizer.transform(self.product_names)
                self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
                self.product_indices = model_data['product_indices']
                self.num_recommendations = model_data['num_recommendations']
                self.model_path = model_path
                print(f"Model loaded successfully from {model_path}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return True

    def get_similar_products(self, product_name):
        # Always compare against the vectorized list
        product_vector = self.vectorizer.transform([product_name])
        self.tfidf_matrix = self.vectorizer.transform(self.product_names)
        similarity_scores = cosine_similarity(product_vector, self.tfidf_matrix).flatten()
        top_indices = similarity_scores.argsort()[::-1][1:self.num_recommendations+1]
        return [(self.product_names[i], round(similarity_scores[i], 3)) for i in top_indices]

def load_data_from_s3():
    bucket_name = "your-recommender-data"
    prefix      = "cleaned/models.csv"

    s3 = boto3.client\
    (
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = os.getenv("us-east-1"),
    )

    obj   = s3.get_object(Bucket=bucket_name, Key=prefix)
    df_s3 = pd.read_csv(io.BytesIO(obj["Body"].read()))

    os.makedirs(os.path.dirname(csv_local_path), exist_ok=True)
    df_s3.to_csv(csv_local_path, index=False)

    return df_s3


def download_joblib_from_s3(bucket_name, s3_key, local_file_path):
    s3 = boto3.client('s3')
    try:
        with open(local_file_path, 'wb') as f:
            s3.download_fileobj(bucket_name, s3_key, f)
            print(f"Successfully downloaded {s3_key} from s3://{bucket_name} to {local_file_path}")
    except Exception as e:
        print(f"Error downloading {s3_key} from s3://{bucket_name}: {e}")
# Download the model from S3
# download_joblib_from_s3("your-recommender-data", "models/cosine_similarity_model.joblib", model_dir + "cosine_similarity_model.joblib")
# model.load_model(model_dir + "cosine_similarity_model.joblib")

@router.get("/top_n_models")
def top_n_models(request: Request,n: int = Query(10, ge=1, le=100) ):
    df= request.app.state.df
    top_n = df.sort_values(by="no_of_ratings", ascending=False).head(n)
    result = top_n[['name', 'no_of_ratings']].to_dict(orient="records")
    return {"top_n_models": result}

@router.get("/similar_products")
def similar_products(request: Request,product_name: str = Query(..., description="Product name")):
    try:
        model=request.app.state.model
        recommendations = model.get_similar_products(product_name)
        return {"similar_products": recommendations}
    except Exception as e:
        return {"error": str(e)}
