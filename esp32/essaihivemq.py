"""
TEST SIMPLE - Connexion à HiveMQ
"""

import paho.mqtt.client as mqtt
import time

# TES identifiants
HOST = "bfa47866fb21414e8e532b321d99dcde.s1.eu.hivemq.cloud"
PORT = 8883
USER = "HUMIYA"
PASS = "bWbiiz2MphgSZ3a"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ CONNECTÉ à HiveMQ !")
    else:
        print(f"❌ ÉCHEC connexion, code: {rc}")
        print("   rc=1 -> Mauvais protocole")
        print("   rc=2 -> Identifiant incorrect")
        print("   rc=3 -> Serveur indisponible")
        print("   rc=4 -> Mauvais user/pass")
        print("   rc=5 -> Non autorisé")

def on_publish(client, userdata, mid):
    print("📤 Message publié avec succès")

# Création client
client = mqtt.Client()
client.username_pw_set(USER, PASS)
client.tls_set()  # SSL nécessaire pour HiveMQ
client.on_connect = on_connect
client.on_publish = on_publish

print("🔌 Tentative de connexion à HiveMQ...")
client.connect(HOST, PORT, 60)
client.loop_start()

time.sleep(2)  # Laisser le temps de connecter

# Publier un message de test
topic = "serre/test"
payload = "Hello from Python!"
result = client.publish(topic, payload)

if result.rc == mqtt.MQTT_ERR_SUCCESS:
    print(f"📤 Publié sur {topic}: {payload}")
else:
    print(f"❌ Erreur publication: {result.rc}")

time.sleep(2)
client.loop_stop()
print("Test terminé")