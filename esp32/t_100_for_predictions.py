import requests

base = "https://actual-reindeer-humiya-11975376.koyeb.app"

print("🔍 VÉRIFICATION CORRIGÉE DES IDS")
print("=" * 50)

# Récupérer l'historique complet
response = requests.get(f"{base}/api/v1/sensors/history?limit=1000")
all_data = response.json()
total = len(all_data)
print(f"📊 Nombre total de mesures reçues: {total}")

if total >= 2:
    first_id = all_data[0].get('id', '?')
    last_id = all_data[-1].get('id', '?')
    print(f"   Premier ID: {first_id}")
    print(f"   Dernier ID: {last_id}")
    if first_id != '?' and last_id != '?':
        print(f"   Écart: {last_id - first_id + 1} mesures")
    else:
        print("   ⚠️ IDs non numériques")

    # Afficher les 5 premiers IDs pour vérifier
    print("\n🔢 5 premiers IDs:")
    for i, m in enumerate(all_data[:5]):
        print(f"   {i + 1}. ID: {m.get('id', '?')} - {m['temperature']}°C")
else:
    print("   ⚠️ Moins de 2 mesures dans la réponse")