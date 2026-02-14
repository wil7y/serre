from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.sql import func
from .database import Base

class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(50), default="esp32_01")
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    temperature = Column(Float)
    humidity = Column(Float)
    light = Column(Float)
    soil_moisture = Column(Float)
    rssi = Column(Integer, nullable=True)

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    temperature_1h = Column(Float)
    humidity_1h = Column(Float)
    light_1h = Column(Float)
    soil_1h = Column(Float)
    temperature_6h = Column(Float)
    humidity_6h = Column(Float)
    light_6h = Column(Float)
    soil_6h = Column(Float)
    model_version = Column(String(20), default="1.0.0")

class ActuatorCommand(Base):
    __tablename__ = "actuator_commands"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    heater = Column(Boolean, default=False)
    pump = Column(Boolean, default=False)
    lights = Column(Boolean, default=False)
    executed = Column(Boolean, default=False)
    executed_at = Column(DateTime(timezone=True), nullable=True)

class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(50), unique=True)
    model_type = Column(String(50))
    training_date = Column(DateTime(timezone=True))
    last_reading_id = Column(Integer)
    samples_count = Column(Integer)
    mse = Column(Float)
    r2 = Column(Float)
    is_active = Column(Boolean, default=True)