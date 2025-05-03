from fastapi import APIRouter, Query, Request
import boto3
import joblib
import os
import numpy as np

router = APIRouter()
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "lightfm_collab_model.joblib")

@router.get("/recommend_products")
def recommend_products(
    request: Request,
    user: str = Query(...),
    top_n: int = Query(20, ge=1, le=100),
):
    # Load model if not loaded
    if not hasattr(request.app.state, "model") or request.app.state.model is None:
        result = load_model(request)
        if isinstance(result, dict) and "error" in result:
            return result  # Return loading error if any

    model = request.app.state.model
    dataset = request.app.state.dataset
    interactions = request.app.state.interactions
    item_mapping = {v: k for k, v in dataset.mapping()[2].items()}
    n_users, n_items = interactions.shape

    if user in dataset.mapping()[0]:
        user_index = dataset.mapping()[0][user]
        scores = model.predict(user_index, np.arange(n_items))
        top_items = np.argsort(-scores)[:top_n]
    else:
        # Fall back to popularity-based recommendations
        item_popularity = interactions.sum(axis=0).A1
        top_items = np.argsort(-item_popularity)[:top_n]

    results = [item_mapping[i] for i in top_items if i in item_mapping]
    return {user: results}

def download_joblib_from_s3(bucket_name, s3_key, local_file_path):
    s3 = boto3.client('s3')
    try:
        with open(local_file_path, 'wb') as f:
            s3.download_fileobj(bucket_name, s3_key, f)
            print(f"Successfully downloaded {s3_key} from s3://{bucket_name} to {local_file_path}")
    except Exception as e:
        print(f"Error downloading {s3_key} from s3://{bucket_name}: {e}")

@router.get("/load_model")
def load_model(request: Request):
    download_joblib_from_s3("your-recommender-data", "models/lightfm_collab_model.joblib", model_path)
    try:
        model_data = joblib.load(model_path)
        request.app.state.model = model_data['model']
        request.app.state.dataset = model_data['dataset']
        request.app.state.interactions = model_data['interactions']
        print(f"✅ Model loaded from {model_path}")
        return {"status": "success"}
    except FileNotFoundError:
        msg = f"❌ Model file not found at {model_path}"
        print(msg)
        return {"error": msg}
    except KeyError as e:
        msg = f"❌ Missing key in model file: {e}"
        print(msg)
        return {"error": msg}
    except Exception as e:
        msg = f"❌ Failed to load model: {e}"
        print(msg)
        return {"error": msg}
