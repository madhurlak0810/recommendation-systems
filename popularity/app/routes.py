from fastapi import APIRouter, Query
import pandas as pd

router = APIRouter()

df = pd.DataFrame()

# return top n models sorted by no_of_ratings.
@router.get("/top_n_models")
def top_n_models(n: int = Query(10, ge=1, le=100)):
    global df

    top_n  = df.sort_values(by="no_of_ratings", ascending=False).head(n)
    result = top_n[['name', 'no_of_ratings']].to_dict(orient="records")

    return {"top_n_models": result}
