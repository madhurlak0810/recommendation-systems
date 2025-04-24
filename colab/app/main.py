from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import router, _train_model_logic


# Global storage
model = None
dataset = None
interactions = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, dataset, interactions

    try:
        # Initialize model at startup
        model, dataset, interactions = _train_model_logic()
        print("✅ Model initialized")
        yield
    finally:
        print("👋 Shutting down app... clean up here if needed")

app = FastAPI(lifespan=lifespan)
app.include_router(router)

# Allow access to df_clean from routes
def get_model():
    return model, dataset, interactions

# Inject dependency into router
router.model = get_model