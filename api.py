from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from predictor import VehiclePredictor

import shutil


app = FastAPI(
    title="Iranian Vehicle Recognition API",
    description="AI API for recognizing Iranian vehicles",
    version="1.0.0",
)

# Model

predictor = VehiclePredictor()

# Static Files


app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)

# Frontend

@app.get("/")
def root():

    return FileResponse(
        "static/index.html"
    )


# Health Check


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
    }



# Prediction


@app.post("/predict")
def predict(
    file: UploadFile = File(...)
):

    temp_path = "temp_image.jpg"

    with open(temp_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer,
        )

    results = predictor.predict(
        temp_path,
        top_k=3,
    )

    return {
        "filename": file.filename,
        "predictions": results,
    }