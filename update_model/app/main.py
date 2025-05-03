from models import ProductHybridModel, ProductCollabModel'
from services.s3_service import download_from_s3
import os

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

download_from_s3("your-recommender-data", "reviews/user_ratings.csv", "data/user_ratings.csv")
download_from_s3("your-recommender-data", "cleaned/models.csv", "data/models.csv")
# hybrid_model = ProductHybridModel("data/user_ratings.csv", "models", n_reviews=10000, n_components=10)
collab_model = ProductCollabModel(csv_path="data/user_ratings.csv", save_model_dir="models", n_components=10, loss='bpr', epochs=30, num_threads=4, k=20)