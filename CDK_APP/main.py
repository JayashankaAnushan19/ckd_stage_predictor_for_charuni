from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import uvicorn
import os
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------
# Load model + encoder
# ---------------------------------------------------
MODEL_PATH = "models/best_model.joblib"
ENCODER_PATH = "models/label_encoder.joblib"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)


# ---------------------------------------------------
# Create API
# ---------------------------------------------------
app = FastAPI()

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_ui():
    """Serve the main HTML UI"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)


@app.post("/predict")
async def predict(request: Request):
    """Handle prediction from frontend"""
    data = await request.json()

    # Extract incoming fields
    age = data["Age"]
    egfr = data["eGFR"]
    album = data["Albuminuria_Level"]
    gender = data["Gender"]
    diabetes = data["Diabetes"]
    hypertension = data["Hypertension"]
    region = data["Region"]

    # categorical encoding
    gender_map = {"Male": 1, "Female": 0}
    region_map = {"Urban": 0, "Rural": 1, "Semi-urban": 2}

    gender_enc = gender_map[gender]
    region_enc = region_map[region]

    # KDIGO fields (required by model)
    kdigo_g = 0
    kdigo_a = 0

    # Build DataFrame in EXACT order
    column_order = [
        'Age', 'eGFR', 'Albuminuria_Level', 'Gender',
        'Diabetes', 'Hypertension', 'Region',
        'KDIGO_G', 'KDIGO_A'
    ]

    features = pd.DataFrame([[
        age, egfr, album,
        gender_enc, diabetes, hypertension,
        region_enc, kdigo_g, kdigo_a
    ]], columns=column_order)

    # Predict
    encoded_pred = model.predict(features)[0]
    decoded_pred = label_encoder.inverse_transform([encoded_pred])[0]

    return JSONResponse({"prediction": decoded_pred})


# ---------------------------------------------------
# Launch when running directly
# ---------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
