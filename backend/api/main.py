from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from .database import engine, Base
from .mqtt_client import mqtt_client
from .routers import sensors, predictions, training

load_dotenv()

# Création des tables
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Au démarrage
    print("🚀 Démarrage de l'API...")
    mqtt_client.start()
    print("✅ API prête!")
    yield
    # À l'arrêt
    print("🛑 Arrêt de l'API")

app = FastAPI(
    title="🌿 Serre Intelligente API",
    description="Backend pour serre connectée avec IA MindSpore",
    version="1.0.0",
    lifespan=lifespan
)

# CORS (pour Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(sensors.router, prefix="/api/v1")
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(training.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "project": "🌿 Serre Intelligente",
        "status": "operational",
        "mqtt": "connected",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}