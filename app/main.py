from fastapi import FastAPI
from app.schemas import ProductInput
from app.model import model, pipeline, feature_columns
from app.utils import align_features

import pandas as pd   # ✅ THIS WAS MISSING

app = FastAPI(title="Anomaly Detection API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: ProductInput):
    # 1. Convert input JSON to DataFrame
    df = pd.DataFrame([payload.dict()])

    # 2. Rename API fields to training column names
    df.rename(columns={
        "Count_Category": "Count Category",
        "Price_In_Dollar": "Price In Dollar",
        "Final_Weights_in_Grams": "Final Weights in Grams"
    }, inplace=True)

    # 3. Feature engineering (same as training)
    df_transformed = pipeline.transform(df)

    # 4. Align feature order
    X = align_features(df_transformed, feature_columns)

    # 5. Prediction
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0].max()

    return {
        "prediction": int(prediction),
        "confidence": float(confidence)
    }
