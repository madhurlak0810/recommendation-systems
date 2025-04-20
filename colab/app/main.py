from fastapi import FastAPI, Query
from contextlib import asynccontextmanager
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

# Global storage
model = None
dataset = None
interactions = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, dataset, interactions

    async def refresh_loop():
        while True:
            try:
                df = pd.read_csv("user_ratings.csv")
                model, dataset, interactions = make_collab_model(df)
                print("✅ Model refreshed")
            except Exception as e:
                print(f"❌ Failed to refresh model: {e}")
            await asyncio.sleep(300)

    # Initial model training
    model, dataset, interactions = make_collab_model(pd.read_csv("user_ratings.csv"))

    # Start the refresh loop
    asyncio.create_task(refresh_loop())

    yield  # Application is running

    # Optional: cleanup code here (on shutdown)

app = FastAPI(lifespan=lifespan)
app.include_router(router)

# Allow access to df_clean from routes
def get_model():
    return model, dataset, interactions

# Inject dependency into router
router.model = get_model