from models import ProductHybridModel, ProductCollabModel

# hybrid_model = ProductHybridModel("data/user_ratings.csv", "models", n_reviews=10000, n_components=10)
collab_model = ProductCollabModel(csv_path="data/user_ratings.csv", save_model_dir="models", n_components=10, loss='bpr', epochs=30, num_threads=4, k=20)