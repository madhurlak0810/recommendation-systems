from fastapi import FastAPI, Query
from typing import Optional
import pandas as pd
import numpy as np
import io
from app.routes import router
import asyncio
from lightfm import LightFM
from lightfm.data import Dataset
import pandas as pd
import numpy as np

def make_collab_model(df):
    # Prepare dataset
    dataset = Dataset()
    dataset.fit(df['user'], df['name'])

    (interactions, weights) = dataset.build_interactions([(row['user'], row['name'], row['rating']) for _, row in df.iterrows()])

    # Train model
    model = LightFM(no_components=10, loss='warp')  # 'warp' good for implicit, use 'mse' or 'logistic' for explicit
    model.fit(interactions, epochs=30, num_threads=4)
    return model, dataset, interactions

app = FastAPI()
app.include_router(router)

# Global storage
model = None
dataset = None
interactions = None

# Load data from S3 on startup
@app.on_event("startup")
async def startup_event():
    global model
    global dataset
    global interactions

    async def refresh_loop():
        while True:
            try:
                model, dataset, interactions = make_collab_model(pd.read_csv("temp.csv"))
                print("Data refreshed")
            except Exception as e:
                print(f"Failed to refresh data: {e}")
            await asyncio.sleep(300)  # refresh every 5 minutes

    model, dataset, interactions = make_collab_model(pd.read_csv("temp.csv"))
    # app.add_event_handler("startup", refresh_loop())
    asyncio.create_task(refresh_loop())


# Allow access to df_clean from routes
def get_model():
    return model, dataset, interactions

# Inject dependency into router
router.model = get_model