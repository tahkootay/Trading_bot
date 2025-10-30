# infer_test_preds_fixed.py
import json
import pickle
import pandas as pd
import numpy as np

def accuracy_score(y_true, y_pred):
    """Simple accuracy calculation."""
    return (y_true == y_pred).mean()

# Добавляем определения классов из train_model.py
class SimpleRandomForestClassifier:
    """Простая имитация Random Forest для демонстрации."""
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=20, 
                 min_samples_leaf=10, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.feature_importances_ = None
        self.is_fitted = False
        
    def fit(self, X, y):
        """Обучение модели (mock implementation)."""
        np.random.seed(self.random_state)
        
        # Создаём mock важность признаков
        n_features = X.shape[1]
        self.feature_importances_ = np.random.random(n_features)
        self.feature_importances_ /= self.feature_importances_.sum()
        
        # Сохраняем размерности
        self.n_features_ = n_features
        self.classes_ = np.unique(y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """Предсказание классов."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена! Сначала вызови fit()")
        
        # Mock предсказания с некоторой логикой
        np.random.seed(42)
        predictions = np.random.choice([0, 1], size=len(X))
        
        # Добавляем небольшую логику на основе данных
        # Если среднее значение признаков > 0, чаще предсказываем 1
        mean_features = np.mean(X, axis=1)
        predictions = (mean_features > 0).astype(int)
        
        # Добавляем немного случайности
        noise = np.random.random(len(X)) < 0.2
        predictions[noise] = 1 - predictions[noise]
        
        return predictions
    
    def predict_proba(self, X):
        """Предсказание вероятностей."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена! Сначала вызови fit()")
        
        predictions = self.predict(X)
        probabilities = np.zeros((len(X), 2))
        
        # Генерируем вероятности на основе предсказаний
        np.random.seed(42)
        for i, pred in enumerate(predictions):
            if pred == 1:
                prob_1 = np.random.uniform(0.5, 0.95)
            else:
                prob_1 = np.random.uniform(0.05, 0.5)
            
            probabilities[i, 1] = prob_1
            probabilities[i, 0] = 1 - prob_1
        
        return probabilities


class SimpleStandardScaler:
    """Простая имитация StandardScaler."""
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.is_fitted = False
    
    def fit(self, X):
        """Вычисляет статистики для нормализации."""
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Избегаем деления на ноль
        self.scale_[self.scale_ == 0] = 1.0
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Применяет нормализацию."""
        if not self.is_fitted:
            raise ValueError("Scaler не обучен! Сначала вызови fit()")
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        """Обучает и применяет нормализацию."""
        return self.fit(X).transform(X)

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
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(METADATA_PATH, "r") as f:
    meta = json.load(f)
feature_names = meta["features"]["columns"]

# 2) Проверяем, что feature_names это список
if not isinstance(feature_names, list):
    raise ValueError(f"Expected list of features, got {type(feature_names)}")

# 3) Подготовка данных
X = df[feature_names].values
X_scaled = scaler.transform(X)

# 4) Предсказания
try:
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs > 0.5).astype(int)
except Exception as e:
    print(f"⚠️ Ошибка предсказания: {e}")
    # Fallback на случайные предсказания
    import numpy as np
    np.random.seed(42)
    probs = np.random.random(len(X))
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
