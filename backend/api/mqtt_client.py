import paho.mqtt.client as mqtt
import json
import os
import numpy as np
from datetime import datetime
from sqlalchemy import desc
from .database import SessionLocal
from .models import SensorReading, Prediction
from .ai_service import ai_service
from .auto_train import continuous_learner
from sqlalchemy import text

# TEST CONNEXION NEON
try:
    db = SessionLocal()


    db.execute(text("SELECT 1"))
    db.close()
    print("✅ Connexion Neon OK (test)")
except Exception as e:
    print(f"❌ Connexion Neon ÉCHOUÉE: {e}")

class HiveMQClient:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.username_pw_set(
            os.getenv("HIVEMQ_USER"),
            os.getenv("HIVEMQ_PASS")
        )
        self.client.tls_set()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        print(f"✅ Connecté à HiveMQ (code: {rc})")
        client.subscribe("serre/sensors/#")

    def _build_features(self, db, latest):
        now = datetime.now()
        features = {
            'temperature_in': latest.temperature,
            'humidity_in': latest.humidity,
            'light_in': latest.light,
            'soil_moisture': latest.soil_moisture,
            'hour_sin': np.sin(2 * np.pi * now.hour / 24),
            'hour_cos': np.cos(2 * np.pi * now.hour / 24),
        }

        last_6 = db.query(SensorReading).order_by(desc(SensorReading.id)).limit(6).all()

        if len(last_6) >= 2:
            features['temperature_in_lag_1'] = last_6[1].temperature
            features['humidity_in_lag_1'] = last_6[1].humidity
            features['light_in_lag_1'] = last_6[1].light
            features['soil_moisture_lag_1'] = last_6[1].soil_moisture

        if len(last_6) >= 3:
            features['temperature_in_lag_2'] = last_6[2].temperature
            features['humidity_in_lag_2'] = last_6[2].humidity

        return features

    def _predict_and_save(self, db, reading):
        print("   🔮 Prédiction IA en cours...")
        features = self._build_features(db, reading)
        pred = ai_service.predict(features)

        if pred:
            prediction_db = Prediction(
                temperature_1h=pred['temperature'],
                humidity_1h=pred['humidity'],
                light_1h=pred['light'],
                soil_1h=pred['soil'],
                model_version="1.0.0"
            )
            db.add(prediction_db)
            db.commit()
            print(f"   ✅ Prédiction sauvegardée")

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload.decode()
        print(f"📥 {topic}: {payload[:100]}...")

        db = SessionLocal()
        try:
            if topic == "serre/sensors/all":
                data = json.loads(payload)
                reading = SensorReading(
                    device_id=data.get("device", "esp32_01"),
                    temperature=data.get("temperature", 0),
                    humidity=data.get("humidity", 0),
                    light=data.get("light", 0),
                    soil_moisture=float(data.get("soil", 0)),  # Force le type float,
                    rssi=data.get("rssi", 0)
                )
                db.add(reading)
                db.commit()
                db.refresh(reading)
                print(f"   ✅ Sauvegardé dans Neon (ID: {reading.id})")

                self._predict_and_save(db, reading)

                if reading.id % 10 == 0:
                    print("🔍 Vérification apprentissage continu...")
                    continuous_learner.check_and_train()
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            db.rollback()
        finally:
            db.close()

    def start(self):
        try:
            self.client.connect(
                os.getenv("HIVEMQ_HOST"),
                int(os.getenv("HIVEMQ_PORT", 8883))
            )
            self.client.loop_start()
            print("✅ Client MQTT démarré")
            return True
        except Exception as e:
            print(f"❌ Erreur démarrage MQTT: {e}")
            return False


mqtt_client = HiveMQClient()