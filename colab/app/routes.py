from fastapi import APIRouter, Query
from typing import Optional
import numpy as np
import pandas as pd

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

    n_users, n_items = interactions.shape
    user_index = dataset.mapping()[0][user]
    item_mapping = {v: k for k, v in dataset.mapping()[2].items()}

    scores = model.predict(user_index, np.arange(n_items))
    top_items = np.argsort(-scores)[:top_n]

    return {user: [item_mapping[i] for i in top_items]}