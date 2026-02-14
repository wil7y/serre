from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from ..database import get_db
from ..models import SensorReading

router = APIRouter(prefix="/sensors", tags=["Capteurs"])

@router.get("/latest")
async def get_latest(db: Session = Depends(get_db)):
    reading = db.query(SensorReading).order_by(desc(SensorReading.id)).first()
    if not reading:
        return {"temperature": 0, "humidity": 0, "light": 0, "soil_moisture": 0}
    return {
        "temperature": reading.temperature,
        "humidity": reading.humidity,
        "light": reading.light,
        "soil_moisture": reading.soil_moisture,
        "timestamp": reading.timestamp,
        "device": reading.device_id
    }

@router.get("/history")
async def get_history(limit: int = Query(24, ge=1, le=168), db: Session = Depends(get_db)):
    readings = db.query(SensorReading).order_by(desc(SensorReading.id)).limit(limit).all()
    return [{
        "id": r.id,
        "timestamp": r.timestamp,
        "temperature": r.temperature,
        "humidity": r.humidity,
        "light": r.light,
        "soil": r.soil_moisture
    } for r in reversed(readings)]