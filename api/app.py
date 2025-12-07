"""
FastAPI backend for MMS Fault Classification.

Provides fast prediction API endpoints for vibration fault detection.
"""

import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Initialize FastAPI app
app = FastAPI(
    title="MMS Fault Classification API",
    description="Industrial vibration fault detection using MiniRocket",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model (loaded once at startup)
predictor = None
CLASS_NAMES = {
    0: "bearing_fault",
    1: "misalignment_fault",
    2: "normal",
    3: "unbalance_fault"
}


@app.on_event("startup")
async def load_model():
    """Load model at startup for fast inference."""
    global predictor

    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from dashboard.utils.predictor import FaultPredictor

    logger.info("Loading MiniRocket model...")
    predictor = FaultPredictor(
        model_path=str(PROJECT_ROOT / "models/minirocket/minirocket_model.pkl"),
        label_encoder_path=str(PROJECT_ROOT / "models/minirocket/label_encoder.pkl"),
        scaler_path=str(PROJECT_ROOT / "models/minirocket/scaler.pkl")
    )
    logger.info("Model loaded and warmed up!")


def _read_array_upload(file_bytes: bytes, filename: str) -> np.ndarray:
    """
    Read uploaded file and return numpy array.

    Supports .npy and .csv formats.
    Expected shape: (1024, 3) or (3, 1024)
    """
    if filename.endswith('.npy'):
        arr = np.load(io.BytesIO(file_bytes))
    elif filename.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(io.BytesIO(file_bytes))
        arr = df.values
    else:
        raise ValueError(f"Unsupported file format: {filename}")

    # Validate and reshape
    if arr.shape == (3, 1024):
        arr = arr.T  # Transpose to (1024, 3)
    elif arr.shape == (1024, 3):
        pass  # Already correct
    elif arr.ndim == 1 and len(arr) == 3072:
        arr = arr.reshape(1024, 3)
    else:
        raise ValueError(f"Invalid shape {arr.shape}. Expected (1024, 3) or (3, 1024)")

    return arr.astype(np.float32)


def _predict(signal: np.ndarray) -> dict:
    """
    Run prediction on signal.

    Args:
        signal: numpy array of shape (1024, 3)

    Returns:
        Dictionary with prediction results
    """
    result = predictor.predict(signal)
    return result


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "MMS Fault Classification API",
        "model": "MiniRocket",
        "accuracy": "99.96%"
    }


@app.get("/health")
async def health():
    """Health check for load balancers."""
    return {"status": "healthy", "model_loaded": predictor is not None}


@app.post("/predict")
async def predict_fault(file: UploadFile = File(...)):
    """
    Predict fault from uploaded vibration data.

    Accepts .npy or .csv files with shape (1024, 3) - 1024 samples, 3 axes (X, Y, Z).

    Returns:
        - predicted_class: Fault type (normal, unbalance_fault, misalignment_fault, bearing_fault)
        - confidence: Prediction confidence (0-1)
        - probabilities: Per-class probabilities
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read file
        file_bytes = await file.read()
        signal = _read_array_upload(file_bytes, file.filename)

        # Predict
        result = _predict(signal)

        return JSONResponse({
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "signal_shape": list(signal.shape)
        })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-array")
async def predict_from_array(data: dict):
    """
    Predict fault from JSON array data.

    Expects:
        {"signal": [[x1, y1, z1], [x2, y2, z2], ...]}  # 1024 samples

    Returns:
        Same as /predict endpoint
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        signal = np.array(data["signal"], dtype=np.float32)

        if signal.shape != (1024, 3):
            raise ValueError(f"Invalid shape {signal.shape}. Expected (1024, 3)")

        result = _predict(signal)

        return JSONResponse({
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"]
        })

    except KeyError:
        raise HTTPException(status_code=400, detail="Missing 'signal' field in request")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def get_model_info():
    """Get model information and metadata."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        info = predictor.get_model_info()
        return JSONResponse(info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes")
async def get_classes():
    """Get available fault classes."""
    return {
        "classes": list(CLASS_NAMES.values()),
        "class_map": CLASS_NAMES
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
