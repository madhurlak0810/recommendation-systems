from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import router, load_model


# Global storage
model = None
dataset = None
interactions = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, dataset, interactions

    try:
        # Initialize model at startup
        model, dataset, interactions = load_model()
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