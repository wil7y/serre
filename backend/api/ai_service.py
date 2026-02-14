import os
import json
import numpy as np
import joblib
import mindspore as ms
from mindspore import nn, Tensor, load_checkpoint, load_param_into_net


class ClimatePredictor(nn.Cell):
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


class MindSporeService:
    def __init__(self, model_dir=None):
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            self.model_dir = os.path.join(base_dir, "ai_models")
        else:
            self.model_dir = model_dir

        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_cols = None
        self.input_dim = None
        self.is_loaded = False
        self.load()

    def load(self):
        try:
            print(f"📁 Chargement modèle depuis: {self.model_dir}")

            with open(os.path.join(self.model_dir, "model_info.json"), 'r') as f:
                info = json.load(f)
                self.feature_cols = info['feature_cols']
                self.input_dim = info['input_dim']

            self.model = ClimatePredictor(self.input_dim)
            param_dict = load_checkpoint(os.path.join(self.model_dir, "climate_model.ckpt"))
            load_param_into_net(self.model, param_dict)
            self.model.set_train(False)

            self.scaler_X = joblib.load(os.path.join(self.model_dir, "scaler_X.pkl"))
            self.scaler_y = joblib.load(os.path.join(self.model_dir, "scaler_y.pkl"))

            self.is_loaded = True
            print(f"✅ Modèle MindSpore chargé")
            print(f"   Features: {len(self.feature_cols)}")
            print(f"   Input dim: {self.input_dim}")
            return True
        except Exception as e:
            print(f"⚠️ Modèle non chargé: {e}")
            print("   L'API fonctionnera sans prédictions")
            self.is_loaded = False
            return False

    def predict(self, features_dict):
        if not self.is_loaded:
            return None
        try:
            feature_vector = [features_dict.get(col, 0.0) for col in self.feature_cols]
            X = np.array(feature_vector).reshape(1, -1)
            X_norm = self.scaler_X.transform(X)
            X_tensor = Tensor(X_norm, ms.float32)
            y_pred_norm = self.model(X_tensor).asnumpy()
            y_pred = self.scaler_y.inverse_transform(y_pred_norm)
            return {
                'temperature': float(y_pred[0, 0]),
                'humidity': float(y_pred[0, 1]),
                'light': float(y_pred[0, 2]),
                'soil': float(y_pred[0, 3])
            }
        except Exception as e:
            print(f"❌ Erreur prédiction: {e}")
            return None


ai_service = MindSporeService()