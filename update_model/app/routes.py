from fastapi import APIRouter, Request

router = APIRouter()

@router.post("/retrain_models")
def retrain_models(request: Request):
    try:
        hybrid = request.app.state.models["hybrid"]
        collab = request.app.state.models["collab"]

        hybrid.train_and_save_model()
        collab.train_and_save_model()

        print("✅ Models retrained via API")
        return {"status": "success", "message": "Models retrained"}
    except Exception as e:
        print(f"❌ Error retraining models: {e}")
        return {"error": str(e)}
