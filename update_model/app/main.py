from models import ProductHybridModel

hybrid_model = ProductHybridModel("data/user_ratings.csv", "models", n_reviews=10000, n_components=10)
