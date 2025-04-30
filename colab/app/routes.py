from fastapi import APIRouter, Query
from typing import Optional
# import boto3
import joblib
import numpy as np
# import pandas as pd
# from io import BytesIO
# from lightfm import LightFM
# from lightfm.data import Dataset
# from lightfm.evaluation import precision_at_k
# from clearml import Task, Logger

router = APIRouter()
router.model = None # This will be injected from main.py

@router.get("/get_products")
def get_popular_products(
    top_n: int = Query(10, ge=1, le=100),
    user: Optional[str] = Query(None),
):
    if router.model is None:
        return {"error": "Model not loaded"}

    model, dataset, interactions = router.model()
    item_meta = joblib.load("item_meta.pkl")

    if user is not None:
        n_users, n_items = interactions.shape
        user_index = dataset.mapping()[0][user]
        item_mapping = {v: k for k, v in dataset.mapping()[2].items()}
        scores = model.predict(user_index, np.arange(n_items))
        top_items = np.argsort(-scores)[:top_n]
    else:
        # new user
        item_popularity = interactions.sum(axis=0).A1  # Sum interactions for each item
        top_items = np.argsort(-item_popularity)[:top_n]

    results = []
    for i in top_items:
        name = item_mapping[i]
        meta = item_meta.get(name, {"main_category": "N/A", "sub_category": "N/A"})
        results.append({
            "name": name,
            "main_category": meta["main_category"],
            "sub_category": meta["sub_category"]
        })

    return {user: results}

def load_model():
    model_path = "../models/lightfm_collab_model.joblib"
    try:
        model_data = joblib.load()
        model = model_data['model']
        dataset = model_data['dataset']
        interactions = model_data['interactions']

        print(f"Collaborative model data loaded successfully from {model_path}")
        return model, dataset, interactions
    except FileNotFoundError:
        print(f"Error: Collaborative model file not found at {model_path}")
        return False
    except KeyError as e:
        print(f"Error: Collaborative model file at {model_path} is missing expected key: {e}")
        return False
    except Exception as e:
        print(f"Error loading collaborative model data from {model_path}: {e}")
        return False

# def _train_model_logic():
#     # retrieve user_ratings
#     bucket_name = "your-recommender-data"
#     prefix = "reviews/user_ratings.csv"
#     s3 = boto3.client("s3")
#     s3_obj = s3.get_object(Bucket=bucket_name, Key=prefix)
#     df = pd.read_csv(BytesIO(s3_obj["Body"].read()))

#     # clearml task setup
#     task: Task = Task.init(
#         project_name="recommendation-systems",
#         task_name="colab model training",
#     )
#     logger = Logger.current_logger()

#     # record parameters
#     params = {
#         "no_components": 10,
#         "loss": "warp",
#         "num_epochs": 30,
#         "k": 5,
#     }
#     task.connect(params)

#     # prepare dataset
#     dataset = Dataset()
#     dataset.fit(df['user'], df['name'])

#     (interactions, weights) = dataset.build_interactions([(row['user'], row['name'], row['rating']) for _, row in df.iterrows()])

#     # train model
#     model = LightFM(no_components=params["no_components"], loss=params["loss"])
#     for epoch in range(params["num_epochs"]):
#         # only fit partial
#         model.fit_partial(interactions, epochs=1, num_threads=4)

#         precision = precision_at_k(model, interactions, k=params["k"]).mean()

#         logger.report_scalar(f"Precision@{params['k']}", "Epoch", iteration=epoch, value=precision)
#         print(f"Epoch {epoch}/{params['num_epochs']} - Precision@{params['k']}: {precision:.4f}")
    
#     # save model artifacts
#     joblib.dump(model, "lightfm_model.pkl")
#     task.upload_artifact(name="lightfm_model", artifact_object="lightfm_model.pkl")
#     joblib.dump(dataset, "dataset.pkl")
#     task.upload_artifact("dataset", artifact_object="dataset.pkl")
#     joblib.dump(interactions, "interactions.pkl")
#     task.upload_artifact("interactions", artifact_object="interactions.pkl")
#     task.close()

#     item_meta = df.drop_duplicates(subset="name").set_index("name")[["main_category", "sub_category"]].to_dict(orient="index")
#     joblib.dump(item_meta, "item_meta.pkl")

#     return model, dataset, interactions

# @router.get("/train_model")
# def train_collab_model():
#     model, dataset, interactions = _train_model_logic()

#     # Optional: update the global reference if needed (e.g. for live reload)
#     router.model = lambda: (model, dataset, interactions)

#     return {"message": "Model trained and artifacts saved successfully."}