# infer_test_preds_fixed.py
import json
import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Добавляем project root в путь для правильных импортов
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def accuracy_score(y_true, y_pred):
    """Simple accuracy calculation."""
    return np.mean(y_true == y_pred)

# Параметры (проверь пути)
TEST_CSV = "data/processed/test.csv"
MODEL_PATH = "models/random_forest/random_forest_v1.pkl"
SCALER_PATH = "models/scalers/scaler_v1.pkl"
METADATA_PATH = "models/metadata/metadata_v1.json"
OUT_CSV = "data/processed/test_with_preds.csv"
THRESH_BUY = 0.6
THRESH_SELL = 0.4

# 1) Загрузка данных и артефактов
df = pd.read_csv(TEST_CSV)

# Импортируем класс модели для корректной десериализации
try:
    from train_model import SimpleRandomForestClassifier, SimpleStandardScaler
except ImportError:
    print("❌ Не удалось импортировать классы из train_model.py")
    print("   Убедитесь, что файл scripts/train_model.py существует")
    sys.exit(1)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(METADATA_PATH, "r") as f:
    meta = json.load(f)

# 2) Правильно извлекаем названия колонок из метаданных
features_info = meta["features"]
if isinstance(features_info, dict) and "columns" in features_info:
    feature_names = features_info["columns"]
else:
    feature_names = features_info

print(f"📊 Загружено признаков: {len(feature_names)}")
print(f"   Первые 5: {feature_names[:5]}")

# 3) Подготовка данных
X = df[feature_names].values
X_scaled = scaler.transform(X)

# 4) Предсказания
probs = model.predict_proba(X_scaled)[:, 1]
preds = (probs > 0.5).astype(int)

# 5) Торговые сигналы
def make_signal(p):
    if p > THRESH_BUY:
        return "Buy"
    elif p < THRESH_SELL:
        return "Sell"
    else:
        return "Hold"

signals = [make_signal(p) for p in probs]

# 6) Формируем результат
out = df.copy()
out["pred"] = preds
out["prob_up"] = probs
out["signal"] = signals
out.to_csv(OUT_CSV, index=False)
print(f"✅ Файл сохранён: {OUT_CSV}")

# 7) Краткая сводка
counts = out["signal"].value_counts().to_dict()
print("Signals count:", counts)

# 8) Точность сигналов, если есть 'target'
if "target" in df.columns:
    results = {}
    for sig in ["Buy", "Sell"]:
        mask = out["signal"] == sig
        if mask.sum() == 0:
            results[sig] = {"count": 0, "accuracy": None}
            continue
        y_true = df.loc[mask, "target"].values
        y_pred = np.array([1 if sig == "Buy" else 0] * len(y_true))
        acc = accuracy_score(y_true, y_pred)
        results[sig] = {"count": int(mask.sum()), "accuracy": float(acc)}
    print("\nSignal accuracies:")
    print(json.dumps(results, indent=2))
else:
    print("⚠️ В test.csv нет столбца 'target' — точность не посчитана.")
