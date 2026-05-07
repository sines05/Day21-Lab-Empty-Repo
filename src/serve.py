from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

GCS_BUCKET = os.environ.get("GCS_BUCKET", "default-bucket")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

def download_model():
    """Tải file model.pkl từ GCS về máy khi server khởi động."""
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH))
        
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_MODEL_KEY)
        blob.download_to_filename(MODEL_PATH)
        print(f"Model downloaded successfully from gs://{GCS_BUCKET}/{GCS_MODEL_KEY} to {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to download model: {e}")

# Gọi hàm này khi module được import (chạy khi server khởi động)
download_model()

# Khởi tạo model nếu file tồn tại
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    """Endpoint kiểm tra sức khỏe server. GitHub Actions dùng endpoint này để xác nhận deploy thành công."""
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luận.
    Đầu vào: JSON {"features": [f1, f2, ..., f12]}
    Đầu ra:  JSON {"prediction": <0|1|2>, "label": <"thấp"|"trung_bình"|"cao">}
    """
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Try restarting the service.")

    preds = model.predict([req.features])
    pred_val = int(preds[0])
    
    label_map = {0: "thấp", 1: "trung_bình", 2: "cao"}
    
    return {"prediction": pred_val, "label": label_map.get(pred_val, "unknown")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
