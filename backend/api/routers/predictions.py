from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc
from ..database import get_db
from ..models import Prediction

router = APIRouter(prefix="/predictions", tags=["Prédictions"])

@router.get("/latest")
async def get_latest_prediction(db: Session = Depends(get_db)):
    pred = db.query(Prediction).order_by(desc(Prediction.id)).first()
    if not pred:
        return {
            "temperature_1h": None,
            "humidity_1h": None,
            "light_1h": None,
            "soil_1h": None,
            "message": "Aucune prédiction disponible"
        }
    return {
        "timestamp": pred.timestamp,
        "temperature_1h": pred.temperature_1h,
        "humidity_1h": pred.humidity_1h,
        "light_1h": pred.light_1h,
        "soil_1h": pred.soil_1h,
        "model_version": pred.model_version
    }

@router.get("/history")
async def get_prediction_history(limit: int = 24, db: Session = Depends(get_db)):
    predictions = db.query(Prediction).order_by(desc(Prediction.id)).limit(limit).all()
    return [{
        "timestamp": p.timestamp,
        "temperature_1h": p.temperature_1h,
        "humidity_1h": p.humidity_1h,
        "light_1h": p.light_1h,
        "soil_1h": p.soil_1h
    } for p in reversed(predictions)]