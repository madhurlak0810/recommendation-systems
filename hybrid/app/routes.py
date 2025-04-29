from fastapi import APIRouter
import requests

router = APIRouter()

COLAB_SERVICE_URL      = "http://colab-service:8000"
POPULARITY_SERVICE_URL = "http://popularity-service:8000"

@router.get("/hybrid_rec")
def hybrid_rec(user_id: int, n: int = 10):
    # call colab API
    try:
        colab_response = requests.get(f"{COLAB_SERVICE_URL}/get_products", params={"n": n, "user_id": user_id})
        colab_results  = colab_response.json().get("recommendations", [])
    except Exception as e:
        colab_results = []

    # call popularity API
    try:
        pop_response = requests.get(f"{POPULARITY_SERVICE_URL}/top_n_models", params={"n": n})
        pop_results  = pop_response.json().get("top_n_models", [])
    except Exception as e:
        pop_results = []

    # Simple hybrid logic (weighted / merging etc.)
    final_recommendations = colab_results + pop_results
    final_recommendations = final_recommendations[:n]

    return {"hybrid_rec": final_recommendations}
