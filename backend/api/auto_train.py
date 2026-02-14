import os
import numpy as np
import pandas as pd
import json
import mindspore as ms
from mindspore import nn, Tensor, context, save_checkpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
from sqlalchemy.orm import Session
from .database import SessionLocal
from .models import SensorReading, ModelMetadata
from .ai_service import ClimatePredictor

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class ContinuousLearner:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.model_dir = os.path.join(base_dir, "ai_models")
        self.min_samples = 100
        self.is_training = False

    def check_and_train(self):
        if self.is_training:
            print("⚠️ Entraînement déjà en cours")
            return

        db = SessionLocal()
        self.is_training = True
        try:
            latest_model = db.query(ModelMetadata).order_by(ModelMetadata.id.desc()).first()
            last_id = latest_model.last_reading_id if latest_model else 0
            new_count = db.query(SensorReading).filter(SensorReading.id > last_id).count()
            print(f"📊 Nouvelles données: {new_count}/{self.min_samples}")

            if new_count >= self.min_samples:
                print("🚀 DÉCLENCHEMENT ENTRAÎNEMENT AUTOMATIQUE")
                self.train_new_model(db, last_id)
        finally:
            self.is_training = False
            db.close()

    def train_new_model(self, db, last_id):
        all_readings = db.query(SensorReading).all()
        if len(all_readings) < 200:
            print("   ⚠️ Pas assez de données totales")
            return

        data = [{
            'temperature_in': r.temperature,
            'humidity_in': r.humidity,
            'light_in': r.light,
            'soil_moisture': r.soil_moisture,
            'timestamp': r.timestamp
        } for r in all_readings]

        df = pd.DataFrame(data)
        df = self.engineer_features(df)
        model, scaler_X, scaler_y, metrics = self.train_mindspore(df)

        # --- MODIFICATION ICI ---
        version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Utiliser un dossier temporaire pour la sauvegarde
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        tmp_ckpt = os.path.join(tmp_dir, f"climate_model_{version}.ckpt")
        tmp_scaler_X = os.path.join(tmp_dir, f"scaler_X_{version}.pkl")
        tmp_scaler_y = os.path.join(tmp_dir, f"scaler_y_{version}.pkl")

        # Sauvegarder dans /tmp
        save_checkpoint(model, tmp_ckpt)
        joblib.dump(scaler_X, tmp_scaler_X)
        joblib.dump(scaler_y, tmp_scaler_y)

        # 2. Mettre à jour le modèle actif en copiant depuis /tmp
        self.update_active_model(tmp_dir, version)
        # --- FIN DE LA MODIFICATION ---

        metadata = ModelMetadata(
            version=version,
            model_type="MindSpore",
            training_date=datetime.now(),
            last_reading_id=db.query(SensorReading).order_by(SensorReading.id.desc()).first().id,
            samples_count=len(df),
            mse=metrics['mse'],
            r2=metrics['r2']
        )
        db.add(metadata)
        db.commit()
        print(f"✅ NOUVEAU MODÈLE ENTRAÎNÉ: v{version}")
        print(f"   R² = {metrics['r2']:.4f}")

    def engineer_features(self, df):
        df = df.sort_values('timestamp')
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        for col in ['temperature_in', 'humidity_in', 'light_in', 'soil_moisture']:
            df[f'{col}_lag_1'] = df[col].shift(1)
            df[f'{col}_lag_2'] = df[col].shift(2)
            df[f'{col}_lag_3'] = df[col].shift(3)
            df[f'{col}_lag_6'] = df[col].shift(6)

        for col in ['temperature_in', 'humidity_in']:
            df[f'{col}_ma_3'] = df[col].rolling(3).mean()
            df[f'{col}_ma_6'] = df[col].rolling(6).mean()

        df['temp_diff'] = df['temperature_in'].diff()
        df['hum_diff'] = df['humidity_in'].diff()
        return df.dropna()

    def train_mindspore(self, df):
        feature_cols = [col for col in df.columns if col not in [
            'timestamp', 'temperature_in', 'humidity_in', 'light_in', 'soil_moisture'
        ]]
        target_cols = ['temperature_in', 'humidity_in', 'light_in', 'soil_moisture']

        X = df[feature_cols].values.astype(np.float32)
        y = df[target_cols].values.astype(np.float32)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_norm = scaler_X.fit_transform(X)
        y_norm = scaler_y.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y_norm, test_size=0.2, random_state=42
        )

        model = ClimatePredictor(input_dim=X.shape[1], hidden_dim=64, output_dim=4)
        loss_fn = nn.MSELoss()
        optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)

        X_train_tensor = Tensor(X_train, ms.float32)
        y_train_tensor = Tensor(y_train, ms.float32)

        def forward_fn(X, y):
            return loss_fn(model(X), y)

        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

        for _ in range(20):
            loss, grads = grad_fn(X_train_tensor, y_train_tensor)
            optimizer(grads)

        model.set_train(False)
        X_test_tensor = Tensor(X_test, ms.float32)
        y_pred_norm = model(X_test_tensor).asnumpy()

        y_test_real = scaler_y.inverse_transform(y_test)
        y_pred_real = scaler_y.inverse_transform(y_pred_norm)

        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test_real, y_pred_real)
        r2 = r2_score(y_test_real, y_pred_real)

        return model, scaler_X, scaler_y, {'mse': mse, 'r2': r2}

    def update_active_model(self, source_dir, version):
        import shutil
        # Copier depuis le dossier source (tmp_dir) vers le dossier de destination (self.model_dir)
        shutil.copy(os.path.join(source_dir, f"climate_model_{version}.ckpt"),
                    os.path.join(self.model_dir, "climate_model.ckpt"))
        shutil.copy(os.path.join(source_dir, f"scaler_X_{version}.pkl"),
                    os.path.join(self.model_dir, "scaler_X.pkl"))
        shutil.copy(os.path.join(source_dir, f"scaler_y_{version}.pkl"),
                    os.path.join(self.model_dir, "scaler_y.pkl"))
        print(f"   ✅ Modèle actif mis à jour → version {version}")


continuous_learner = ContinuousLearner()