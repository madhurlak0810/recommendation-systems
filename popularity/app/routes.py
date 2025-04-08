from fastapi import APIRouter, Query
from typing import Optional
import pandas as pd

router = APIRouter()
df_loader = None  # This will be injected from main.py

@router.get("/popular-products")
def get_popular_products(
    top_n: int = Query(10, ge=1, le=100),
    main_category: Optional[str] = Query(None),
    sub_category: Optional[str] = Query(None)
):
    if df_loader is None:
        return {"error": "Data not loaded"}

    df = df_loader().copy()

    if main_category:
        df = df[df['main_category'].str.lower() == main_category.lower()]
    if sub_category:
        df = df[df['sub_category'].str.lower() == sub_category.lower()]

    top_products = (
        df.sort_values("popularity_score", ascending=False)
        .head(top_n)
        .loc[:, ["name", "main_category", "sub_category", "ratings", "no_of_ratings", "popularity_score"]]
        .to_dict(orient="records")
    )

    return {"results": top_products}