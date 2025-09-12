
from fastapi import FastAPI, HTTPException, Header, Request, Response
from pydantic import BaseModel
import joblib, os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

MODEL_PATH = os.environ.get("MODEL_PATH", "./outputs/model.pkl")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "change-me-secure-token")

app = FastAPI(title="Accident Severity Inference API")
model = None
def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model

class PredictRequest(BaseModel):
    features: dict

REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['endpoint','method','status'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency', ['endpoint'])

@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/health")
def health():
    REQUEST_COUNT.labels(endpoint="/health", method="GET", status="200").inc()
    return {"status":"ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    start = time.time()
    clf = load_model()
    X = pd.DataFrame([req.features])
    pred = clf.predict(X)[0]
    duration = time.time() - start
    REQUEST_LATENCY.labels(endpoint="/predict").observe(duration)
    REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="200").inc()
    return {"prediction": int(pred), "duration": duration}

@app.post("/admin/register")
def admin_register(payload: dict, x_admin_token: str = Header(None)):
    if x_admin_token != ADMIN_TOKEN:
        REQUEST_COUNT.labels(endpoint="/admin/register", method="POST", status="403").inc()
        raise HTTPException(status_code=403, detail="Forbidden")
    return {"status":"ok", "message":"admin action performed"}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
