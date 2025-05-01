from fastapi import APIRouter, Query
import pandas as pd
import boto3
import io
import os
import joblib

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter()

csv_local_path = "data/models.csv"
model_dir      = "model/"

df = pd.DataFrame()

class ProductNameSimilarityModel:
    def __init__(self, csv_path, save_model_dir, num_recommendations=10, sample_size=10000):
        self.product_names       = None
        self.vectorizer          = None
        self.tfidf_matrix        = None
        self.cosine_sim          = None
        self.product_indices     = None
        self.num_recommendations = None

        self.csv_path            = csv_path
        self.save_model_dir      = save_model_dir
        self.sample_size         = sample_size
        self.model_path          = os.path.join(self.save_model_dir, 'cosine_similarity_model.joblib')
        self.num_recommendations = num_recommendations
        self.cosinesimilarproducts()

    def cosinesimilarproducts(self):
        try:
            df = pd.read_csv(self.csv_path)
            if len(df) > self.sample_size:
                df = df.sample(n=self.sample_size, random_state=42)

            self.product_names   = pd.Series(df['name']).dropna().unique()
            self.vectorizer      = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix    = self.vectorizer.fit_transform(self.product_names)
            self.cosine_sim      = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            self.product_indices = {name: idx for idx, name in enumerate(self.product_names)}

            os.makedirs(self.save_model_dir, exist_ok=True)
            joblib.dump({
                'product_names': self.product_names,
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'cosine_sim': self.cosine_sim,
                'product_indices': self.product_indices,
                'num_recommendations': self.num_recommendations
            }, self.model_path)
            print(f"Model data saved to {self.model_path}")
            return True
        except Exception as e:
            print(f"Error training or saving model: {e}")
            return False

    def load_model(self, model_path):
        try:
            model_data               = joblib.load(model_path)
            self.product_names       = model_data['product_names']
            self.vectorizer          = model_data['vectorizer']
            self.tfidf_matrix        = model_data['tfidf_matrix']
            self.cosine_sim          = model_data['cosine_sim']
            self.product_indices     = model_data['product_indices']
            self.num_recommendations = model_data['num_recommendations']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_similar_products(self, product_name):
        product_vector    = self.vectorizer.transform([product_name])
        similarity_scores = cosine_similarity(product_vector, self.tfidf_matrix).flatten()
        top_indices       = similarity_scores.argsort()[::-1][1:self.num_recommendations + 1]
        return [(self.product_names[i], round(similarity_scores[i], 3)) for i in top_indices]

def load_data_from_s3():
    load_dotenv()
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

def set_global_df(new_df):
    global df
    df.drop(df.index, inplace=True)
    df.update(new_df)

# Load data and model at startup
set_global_df(load_data_from_s3())
model = ProductNameSimilarityModel(csv_path=csv_local_path, save_model_dir=model_dir)

@router.get("/top_n_models")
def top_n_models(n: int = Query(10, ge=1, le=100)):
    top_n = df.sort_values(by="no_of_ratings", ascending=False).head(n)
    result = top_n[['name', 'no_of_ratings']].to_dict(orient="records")
    return {"top_n_models": result}

@router.get("/similar_products")
def similar_products(product_name: str = Query(..., description="Product name")):
    try:
        recommendations = model.get_similar_products(product_name)
        return {"similar_products": recommendations}
    except Exception as e:
        return {"error": str(e)}
