/*
=============================================
   🌿 SERRE INTELLIGENTE - ESP32
   Version PRODUCTION - Capteurs réels
   WiFiManager + HiveMQ + Actionneurs
=============================================
*/

#include <WiFi.h>
#include <WiFiManager.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <BH1750.h>
#include <Wire.h>

// PINS
#define DHTPIN 4
#define DHTTYPE DHT22
#define SOIL_PIN 34
#define HEATER_PIN 15
#define PUMP_PIN 2
#define LIGHTS_PIN 4

// OBJETS
DHT dht(DHTPIN, DHTTYPE);
BH1750 lightMeter;
WiFiClientSecure espClient;
PubSubClient client(espClient);

// HIVEMQ (FIXE)
const char* mqtt_server = "bfa47866fb21414e8e532b321d99dcde.s1.eu.hivemq.cloud";
const int mqtt_port = 8883;
const char* mqtt_user = "HUMIYA";
const char* mqtt_pass = "bWbiiz2MphgSZ3a";
const char* device_id = "esp32_01";

// TIMING
unsigned long lastSend = 0;
unsigned long lastHeartbeat = 0;
const long SEND_INTERVAL = 30000;
const long HEARTBEAT_INTERVAL = 60000;

// ÉTAT
bool heaterState = false;
bool pumpState = false;
bool lightsState = false;

// ===== WIFI MANAGER =====
void setup_wifi() {
  Serial.println("\n📶 WiFiManager - Mode configuration");
  WiFiManager wm;
  wm.setConfigPortalTimeout(180);

  if (!wm.autoConnect("Serre-ESP32", "serre1234")) {
    Serial.println("❌ Échec WiFiManager");
    ESP.restart();
  }
  Serial.println("✅ WiFi connecté!");
  Serial.print("   IP: ");
  Serial.println(WiFi.localIP());
}

// ===== CONNEXION HIVEMQ =====
void reconnect() {
  int attempts = 0;
  while (!client.connected() && attempts < 5) {
    Serial.print("📡 Connexion HiveMQ...");
    espClient.setInsecure();

    if (client.connect(device_id, mqtt_user, mqtt_pass)) {
      Serial.println("✅ Connecté!");
      client.subscribe("serre/commandes/#");
      client.publish("serre/status", "online");
    } else {
      Serial.print("❌ Échec, rc=");
      Serial.print(client.state());
      Serial.println(" (nouvelle tentative dans 5s)");
      attempts++;
      delay(5000);
    }
  }
  if (!client.connected()) ESP.restart();
}

// ===== LECTURE CAPTEURS =====
float read_temperature() {
  float t = dht.readTemperature();
  return isnan(t) ? 0.0 : t;
}

float read_humidity() {
  float h = dht.readHumidity();
  return isnan(h) ? 0.0 : h;
}

float read_light() {
  float lux = lightMeter.readLightLevel();
  return (isnan(lux) || lux < 0) ? 0.0 : lux;
}

float read_soil() {
  int raw = analogRead(SOIL_PIN);
  float percent = map(raw, 1800, 3200, 0, 100);
  return constrain(percent, 0, 100);
}

// ===== CONTROLE ACTIONNEURS =====
void control_heater(bool state) {
  heaterState = state;
  digitalWrite(HEATER_PIN, state ? HIGH : LOW);
  client.publish("serre/status/heater", state ? "1" : "0");
}

void control_pump(bool state) {
  pumpState = state;
  digitalWrite(PUMP_PIN, state ? HIGH : LOW);
  client.publish("serre/status/pump", state ? "1" : "0");
}

void control_lights(bool state) {
  lightsState = state;
  digitalWrite(LIGHTS_PIN, state ? HIGH : LOW);
  client.publish("serre/status/lights", state ? "1" : "0");
}

// ===== RÉCEPTION COMMANDES =====
void callback(char* topic, byte* payload, unsigned int length) {
  Serial.printf("\n📥 Commande: %s\n", topic);
  String message;
  for (int i = 0; i < length; i++) message += (char)payload[i];
  bool state = (message == "1" || message == "true" || message == "ON");

  if (strcmp(topic, "serre/commandes/heater") == 0) control_heater(state);
  else if (strcmp(topic, "serre/commandes/pump") == 0) control_pump(state);
  else if (strcmp(topic, "serre/commandes/lights") == 0) control_lights(state);
}

// ===== ENVOI DONNÉES =====
void send_sensor_data() {
  Serial.println("\n📤 Lecture capteurs...");

  float temp = read_temperature();
  float hum = read_humidity();
  float light = read_light();
  float soil = read_soil();

  Serial.printf("   🌡️ %.1f°C  💧 %.1f%%  💡 %.0f lux  🌱 %.1f%%\n",
                temp, hum, light, soil);

  if (!client.connected()) reconnect();

  client.publish("serre/sensors/temperature", String(temp).c_str());
  client.publish("serre/sensors/humidity", String(hum).c_str());
  client.publish("serre/sensors/light", String(light).c_str());
  client.publish("serre/sensors/soil", String(soil).c_str());

  StaticJsonDocument<384> doc;
  doc["device"] = device_id;
  doc["temperature"] = temp;
  doc["humidity"] = hum;
  doc["light"] = light;
  doc["soil"] = soil;
  doc["rssi"] = WiFi.RSSI();
  doc["heater"] = heaterState;
  doc["pump"] = pumpState;
  doc["lights"] = lightsState;

  char buffer[384];
  serializeJson(doc, buffer);
  client.publish("serre/sensors/all", buffer);
  Serial.println("   ✅ Données envoyées à HiveMQ!");
}

// ===== HEARTBEAT =====
void send_heartbeat() {
  if (!client.connected()) return;
  StaticJsonDocument<128> doc;
  doc["device"] = device_id;
  doc["status"] = "online";
  doc["uptime"] = millis() / 1000;
  doc["heap"] = ESP.getFreeHeap();
  doc["rssi"] = WiFi.RSSI();
  char buffer[128];
  serializeJson(doc, buffer);
  client.publish("serre/heartbeat", buffer);
}

// ===== SETUP =====
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=============================================");
  Serial.println("🚀 SERRE INTELLIGENTE - MODE PRODUCTION");
  Serial.println("=============================================");

  Wire.begin();
  dht.begin();
  lightMeter.begin();

  pinMode(HEATER_PIN, OUTPUT);
  pinMode(PUMP_PIN, OUTPUT);
  pinMode(LIGHTS_PIN, OUTPUT);
  digitalWrite(HEATER_PIN, LOW);
  digitalWrite(PUMP_PIN, LOW);
  digitalWrite(LIGHTS_PIN, LOW);

  setup_wifi();

  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
  client.setKeepAlive(60);
  client.setBufferSize(1024);

  reconnect();

  Serial.println("\n✅ ESP32 PRÊT !");
  Serial.println("   📤 Envoi capteurs : 30s");
}

// ===== LOOP =====
void loop() {
  if (!client.connected()) reconnect();
  client.loop();

  unsigned long now = millis();
  if (now - lastSend > SEND_INTERVAL) {
    lastSend = now;
    send_sensor_data();
  }

  if (now - lastHeartbeat > HEARTBEAT_INTERVAL) {
    lastHeartbeat = now;
    send_heartbeat();
  }
}