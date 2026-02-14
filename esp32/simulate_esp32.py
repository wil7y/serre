"""
🔧 SIMULATEUR ESP32 - Envoie de fausses données à HiveMQ
Pour tester sans matériel
"""

import paho.mqtt.client as mqtt
import json
import time
import random
from datetime import datetime

# Configuration HiveMQ (TES identifiants)
HOST = "bfa47866fb21414e8e532b321d99dcde.s1.eu.hivemq.cloud"
PORT = 8883
USER = "HUMIYA"
PASS = "bWbiiz2MphgSZ3a"


def on_connect(client, userdata, flags, rc):
    print(f"✅ Connecté à HiveMQ (code: {rc})")


# Client MQTT
client = mqtt.Client()
client.username_pw_set(USER, PASS)
client.tls_set()
client.on_connect = on_connect

client.connect(HOST, PORT)
client.loop_start()

print("🚀 Simulation ESP32 - Envoi toutes les 10 secondes")
print("=" * 50)

try:
    while True:
        # Générer des données aléatoires réalistes
        data = {
            "device": "esp32_simule",
            "temperature": round(20 + random.random() * 5, 1),
            "humidity": round(60 + random.random() * 10, 1),
            "light": int(10000 + random.random() * 5000),
            "soil": round(50 + random.random() * 10, 1),
            "rssi": -random.randint(40, 70),
            "timestamp": datetime.now().isoformat()
        }

        # Publier
        payload = json.dumps(data)
        client.publish("serre/sensors/all", payload)

        print(f"📤 Envoyé: 🌡️ {data['temperature']}°C  💧 {data['humidity']}%")

        time.sleep(10)  # Toutes les 10 secondes

except KeyboardInterrupt:
    print("\n🛑 Arrêt simulation")
    client.loop_stop()