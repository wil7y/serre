from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import ModelMetadata
from ..auto_train import continuous_learner

router = APIRouter(prefix="/training", tags=["Apprentissage"])

@router.get("/status")
async def get_training_status(db: Session = Depends(get_db)):
    latest = db.query(ModelMetadata).order_by(ModelMetadata.id.desc()).first()
    if latest:
        return {
            "status": "active",
            "last_training": latest.training_date,
            "version": latest.version,
            "samples": latest.samples_count,
            "r2_score": latest.r2,
            "mse": latest.mse
        }
    return {
        "status": "waiting",
        "message": "En attente de données suffisantes"
    }

@router.post("/trigger")
async def trigger_training():
    continuous_learner.check_and_train()
    return {"status": "training_started"}