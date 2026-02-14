#!/usr/bin/env python3
"""
🚀 ENTRAÎNEMENT MINDSPORE - DEPUIS NEON
Utilise les VRAIES données de la base de données
Sauvegarde les fichiers dans backend/ai_models/

Auteur: Équipe Serre Intelligente
Date: 2026
Version: 2.0
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import mindspore as ms
from mindspore import nn, Tensor, context, save_checkpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sqlalchemy import create_engine, text
from datetime import datetime
from dotenv import load_dotenv

# ============================================
# CONFIGURATION
# ============================================
# Déterminer le chemin absolu du projet
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
AI_MODELS_DIR = os.path.join(BACKEND_DIR, "ai_models")
ENV_FILE = os.path.join(BACKEND_DIR, ".env")

print("=" * 70)
print("🌿 ENTRAÎNEMENT DU MODÈLE MINDSPORE - SERRE INTELLIGENTE")
print("=" * 70)
print(f"📁 Racine du projet: {PROJECT_ROOT}")
print(f"📁 Dossier de sauvegarde: {AI_MODELS_DIR}")
print(f"📁 Fichier .env: {ENV_FILE}")

# ============================================
# CONFIGURATION MINDSPORE
# ============================================
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
print(f"🧠 MindSpore version: {ms.__version__}")

# ============================================
# 1. CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# ============================================
print("\n📂 Chargement des variables d'environnement...")

if not os.path.exists(ENV_FILE):
    print(f"❌ Fichier .env non trouvé: {ENV_FILE}")
    print("   Vérifie que backend/.env existe")
    sys.exit(1)

load_dotenv(ENV_FILE)
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ DATABASE_URL non trouvée dans .env")
    print("   Vérifie le contenu de backend/.env")
    sys.exit(1)

print(f"✅ DATABASE_URL chargée: {DATABASE_URL[:50]}...")

# ============================================
# 2. CONNEXION À NEON
# ============================================
print("\n📂 Connexion à Neon...")

try:
    engine = create_engine(DATABASE_URL)
    # Test de connexion
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("✅ Connexion à Neon réussie")
except Exception as e:
    print(f"❌ Erreur de connexion à Neon: {e}")
    sys.exit(1)

# ============================================
# 3. CHARGEMENT DES DONNÉES
# ============================================
print("\n📊 Chargement des données depuis Neon...")

query = """
SELECT 
    temperature, 
    humidity, 
    light, 
    soil_moisture,
    timestamp
FROM sensor_readings 
WHERE timestamp > NOW() - INTERVAL '30 days'
ORDER BY timestamp
"""

try:
    df = pd.read_sql(query, engine)
    print(f"   ✅ {len(df)} mesures chargées")
except Exception as e:
    print(f"❌ Erreur lors du chargement des données: {e}")
    sys.exit(1)

if len(df) < 100:
    print("   ⚠️ Pas assez de données ! Minimum requis: 100 mesures")
    print(f"   Actuellement: {len(df)} mesures")
    print("   Laisse l'ESP32 ou le simulateur tourner plus longtemps.")
    sys.exit(1)

print(f"   ✅ Période couverte: {df['timestamp'].min()} à {df['timestamp'].max()}")

# ============================================
# 4. CRÉATION DES FEATURES
# ============================================
print("\n🔧 Création des features...")

df = df.sort_values('timestamp')

# Features temporelles
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Lags (t-1, t-2, t-3, t-6)
for col in ['temperature', 'humidity', 'light', 'soil_moisture']:
    df[f'{col}_lag_1'] = df[col].shift(1)
    df[f'{col}_lag_2'] = df[col].shift(2)
    df[f'{col}_lag_3'] = df[col].shift(3)
    df[f'{col}_lag_6'] = df[col].shift(6)

# Moyennes mobiles
for col in ['temperature', 'humidity']:
    df[f'{col}_ma_3'] = df[col].rolling(3).mean()
    df[f'{col}_ma_6'] = df[col].rolling(6).mean()

# Différences
df['temp_diff'] = df['temperature'].diff()
df['hum_diff'] = df['humidity'].diff()

# Supprimer les NaN
df = df.dropna()
print(f"   ✅ {len(df)} échantillons après création des features")

# ============================================
# 5. PRÉPARATION X et y
# ============================================
target_cols = ['temperature', 'humidity', 'light', 'soil_moisture']
feature_cols = [col for col in df.columns if col not in target_cols + ['timestamp', 'hour']]

X = df[feature_cols].values.astype(np.float32)
y = df[target_cols].values.astype(np.float32)

print(f"   ✅ Features: {len(feature_cols)}")
print(f"   ✅ Targets: {len(target_cols)}")
print(f"   ✅ Forme X: {X.shape}")
print(f"   ✅ Forme y: {y.shape}")

# ============================================
# 6. NORMALISATION
# ============================================
print("\n📊 Normalisation des données...")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_norm = scaler_X.fit_transform(X)
y_norm = scaler_y.fit_transform(y)

print(f"   ✅ X normalisé: μ≈0, σ≈1")
print(f"   ✅ y normalisé: μ≈0, σ≈1")

# ============================================
# 7. DÉFINITION DU MODÈLE
# ============================================
print("\n🧠 Création du modèle MindSpore...")


class ClimatePredictor(nn.Cell):
    """
    Réseau de neurones pour la prédiction climatique
    Architecture: Dense + BatchNorm + ReLU + Dense + BatchNorm + ReLU + Dense
    """

    def __init__(self, input_dim, hidden_dim=64, output_dim=4):
        super().__init__()
        self.fc1 = nn.Dense(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Dense(hidden_dim // 2, output_dim)

    def construct(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


model = ClimatePredictor(input_dim=X.shape[1])
print(f"   ✅ Architecture: {X.shape[1]} → 64 → 32 → 4")

# ============================================
# 8. ENTRAÎNEMENT
# ============================================
print("\n🚀 Entraînement MindSpore...")

loss_fn = nn.MSELoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)

X_tensor = Tensor(X_norm, ms.float32)
y_tensor = Tensor(y_norm, ms.float32)


def forward_fn(X, y):
    return loss_fn(model(X), y)


grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

epochs = 50
losses = []

for epoch in range(epochs):
    loss, grads = grad_fn(X_tensor, y_tensor)
    optimizer(grads)
    loss_value = loss.asnumpy()
    losses.append(loss_value)

    if (epoch + 1) % 10 == 0:
        print(f"   Époque {epoch + 1:2d}/{epochs} - Loss: {loss_value:.6f}")

print(f"   ✅ Entraînement terminé. Loss finale: {losses[-1]:.6f}")

# ============================================
# 9. ÉVALUATION
# ============================================
print("\n📊 Évaluation du modèle...")

X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y_norm, test_size=0.2, random_state=42
)

model.set_train(False)
X_test_tensor = Tensor(X_test, ms.float32)
y_pred_norm = model(X_test_tensor).asnumpy()

# Dénormalisation
y_test_real = scaler_y.inverse_transform(y_test)
y_pred_real = scaler_y.inverse_transform(y_pred_norm)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

mse = mean_squared_error(y_test_real, y_pred_real)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_real, y_pred_real)
r2 = r2_score(y_test_real, y_pred_real)

print(f"   📉 MSE:  {mse:.4f}")
print(f"   📊 RMSE: {rmse:.4f}")
print(f"   📏 MAE:  {mae:.4f}")
print(f"   🎯 R²:   {r2:.4f}")

# Détail par target
print("\n📈 Détail par variable:")
target_names = ['🌡️ Température', '💧 Humidité', '💡 Lumière', '🌱 Sol']
for i, name in enumerate(target_names):
    mse_i = mean_squared_error(y_test_real[:, i], y_pred_real[:, i])
    print(f"   {name}: MSE={mse_i:.4f}")

# ============================================
# 10. SAUVEGARDE DES FICHIERS
# ============================================
print("\n💾 Sauvegarde des fichiers...")

# Créer le dossier s'il n'existe pas
os.makedirs(AI_MODELS_DIR, exist_ok=True)
print(f"   ✅ Dossier vérifié: {AI_MODELS_DIR}")

# Sauvegarde du modèle MindSpore
model_path = os.path.join(AI_MODELS_DIR, "climate_model.ckpt")
save_checkpoint(model, model_path)
print(f"   ✅ Modèle: {model_path}")

# Sauvegarde des scalers
scaler_X_path = os.path.join(AI_MODELS_DIR, "scaler_X.pkl")
scaler_y_path = os.path.join(AI_MODELS_DIR, "scaler_y.pkl")
joblib.dump(scaler_X, scaler_X_path)
joblib.dump(scaler_y, scaler_y_path)
print(f"   ✅ Scalers: scaler_X.pkl, scaler_y.pkl")

# Sauvegarde des informations du modèle
model_info = {
    'feature_cols': feature_cols,
    'target_cols': target_cols,
    'input_dim': X.shape[1],
    'hidden_dim': 64,
    'output_dim': 4,
    'model_type': 'MindSpore',
    'samples': len(X),
    'mse': float(mse),
    'rmse': float(rmse),
    'mae': float(mae),
    'r2': float(r2),
    'training_date': datetime.now().isoformat(),
    'data_start': df['timestamp'].min().isoformat() if len(df) > 0 else None,
    'data_end': df['timestamp'].max().isoformat() if len(df) > 0 else None
}

info_path = os.path.join(AI_MODELS_DIR, "model_info.json")
with open(info_path, 'w', encoding='utf-8') as f:
    json.dump(model_info, f, indent=2, ensure_ascii=False)
print(f"   ✅ Configuration: {info_path}")

# ============================================
# 11. VÉRIFICATION FINALE
# ============================================
print("\n🔍 Vérification des fichiers sauvegardés:")
files = os.listdir(AI_MODELS_DIR)
for f in files:
    size = os.path.getsize(os.path.join(AI_MODELS_DIR, f))
    print(f"   📄 {f:25} ({size / 1024:.1f} Ko)")

# ============================================
# 12. TEST RAPIDE DE PRÉDICTION
# ============================================
print("\n🔮 Test de prédiction avec le modèle entraîné...")

# Prendre un échantillon de test
sample = X[0:1]
sample_norm = scaler_X.transform(sample)
sample_tensor = Tensor(sample_norm, ms.float32)
pred_norm = model(sample_tensor).asnumpy()
pred = scaler_y.inverse_transform(pred_norm)

print(f"   🌡️ Température prédite: {pred[0, 0]:.1f}°C")
print(f"   💧 Humidité prédite: {pred[0, 1]:.1f}%")
print(f"   💡 Lumière prédite: {pred[0, 2]:.0f} lux")
print(f"   🌱 Sol prédit: {pred[0, 3]:.1f}%")

# ============================================
# FIN
# ============================================
print("\n" + "=" * 70)
print("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
print("=" * 70)
print(f"\n📁 Modèle prêt dans: {AI_MODELS_DIR}")
print("\n🚀 Tu peux maintenant :")
print("   1. Relancer l'API: uvicorn backend.api.main:app --reload")
print("   2. Vérifier les prédictions: http://localhost:8000/api/v1/predictions/latest")
print("   3. Déployer sur Koyeb (git push)")
print("=" * 70)