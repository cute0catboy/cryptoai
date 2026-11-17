from pathlib import Path
import features_create as fc
import mpl_plots as mpl
import ai_model as ai
import numpy as np
import pandas as pd
import data_get
from sklearn.metrics import accuracy_score, classification_report

PLOTS = Path(__file__).parent.resolve() / "plots"
PLOTS.mkdir(exist_ok=True)
DATA = Path(__file__).parent.resolve() / "data"
DATA.mkdir(exist_ok=True)

data = data_get.get_data()

# Формирование целевой переменной (y_raw)
next_close = data["close"].shift(-1)
y_raw = (next_close > data["close"]).astype(float)
y_raw[next_close.isna()] = np.nan

features = fc.FeatureCreator(data)

# Доходности (returns)
ret1 = features.ret(1)
ret4 = features.ret(4)
ret12 = features.ret(12)
ret24 = features.ret(24)

# Волатильность (volatilities)
volatility6 = features.volatility(6)
volatility24 = features.volatility(24)
volatility72 = features.volatility(72)

# Относительный объем (volume ratio)
vol_rat24 = features.vol_ratio(24)

# Позиция цены относительно SMA
ptc24 = features.price_to_sma(24)

# Свечные и объемные признаки
hl_range = (data["high"] - data["low"]) / data["close"]
body = (data["close"] - data["open"]) / data["open"]
vol_change = data["volume"].pct_change(1)

data_new = (
    pd.DataFrame({
        # returns
        "ret_1": ret1,
        "ret_4": ret4,
        "ret_12": ret12,
        "ret_24": ret24,
        # candle
        "hl_range": hl_range,
        "body": body,
        # volatilities
        "volatility_6": volatility6,
        "volatility_24": volatility24,
        "volatility_72": volatility72,
        # volume
        "volume_change": vol_change,
        "volume_ratio_24": vol_rat24,
        # price position
        "price_to_sma_24": ptc24,
        # target
        "y": y_raw,
    })
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

print("=" * 60)
print("1. ОБЗОР ДАТАСЕТА ПОСЛЕ ОЧИСТКИ")
print("=" * 60)
print(
    f"Размерность (строк, колонок): {data_new.shape[0]} x {data_new.shape[1]}"
)
print(f"Период данных: с {data_new.index.min()} по {data_new.index.max()}")
print("\nПример первой валидной строки (значения признаков):")
print(data_new.iloc[0].round(5).to_string())

X = data_new[[
    "ret_1",
    "ret_4",
    "ret_12",
    "ret_24",
    "hl_range",
    "body",
    "volatility_6",
    "volatility_24",
    "volatility_72",
    "volume_change",
    "volume_ratio_24",
    "price_to_sma_24",
]]
y = data_new["y"].astype(int)
assert not X.isna().any().any(), "Обнаружены NaN в матрице признаков X!"

# Разбиение выборки по времени (70% / 15% / 15%)
n_samples = X.shape[0]
split_70 = int(n_samples * 0.7)
split_85 = int(n_samples * 0.85)

X_train, y_train = X.iloc[:split_70], y.iloc[:split_70]
X_validation, y_validation = (
    X.iloc[split_70:split_85],
    y.iloc[split_70:split_85],
)
X_test, y_test = X.iloc[split_85:], y.iloc[split_85:]

print("\n" + "=" * 60)
print("2. БАЛАНС КЛАССОВ И РАЗМЕРЫ ВЫБОРОК")
print("=" * 60)
print(
    f"Train      | Строк: {len(y_train):3d} | Доля роста (y=1):"
    f" {y_train.mean():.2%}"
)
print(
    f"Validation | Строк: {len(y_validation):3d} | Доля роста (y=1):"
    f" {y_validation.mean():.2%}"
)
print(
    f"Test       | Строк: {len(y_test):3d} | Доля роста (y=1):"
    f" {y_test.mean():.2%}"
)

assert X_train.shape[0] == y_train.shape[0]
assert X_validation.shape[0] == y_validation.shape[0]
assert X_test.shape[0] == y_test.shape[0]
assert pd.concat([X_train, X_validation, X_test]).equals(X)
assert pd.concat([y_train, y_validation, y_test]).equals(y)

# Клиппинг выбросов по правилу 3 сигм (параметры только с Train)
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)
clip_low = mu - 3 * sd
clip_high = mu + 3 * sd

X_train_clipped = X_train.clip(lower=clip_low, upper=clip_high, axis=1)
X_validation_clipped = X_validation.clip(
    lower=clip_low, upper=clip_high, axis=1
)
X_test_clipped = X_test.clip(lower=clip_low, upper=clip_high, axis=1)

print("\n" + "=" * 60)
print("3. ПРОВЕРКА ДАННЫХ И КЛИППИНГ ВЫБРОСОВ (3 SIGMA)")
print("=" * 60)

# Сводная таблица диагностики Train-выборки
diag_df = pd.DataFrame({
    "Unique_vals": X_train.nunique(),
    "Inf_count": np.isinf(X_train).sum(),
    "Clip_low": clip_low,
    "Clip_high": clip_high,
    "Train_clipped_cnt": (X_train != X_train_clipped).sum(),
    "Train_clipped_%": ((X_train != X_train_clipped).mean() * 100).round(1),
    "Val_clipped_%": (
        (X_validation != X_validation_clipped).mean() * 100
    ).round(1),
    "Test_clipped_%": ((X_test != X_test_clipped).mean() * 100).round(1),
})
print(diag_df.to_string())

# Min-Max нормализация (параметры только с Train)
X_min = X_train_clipped.min(axis=0)
X_max = X_train_clipped.max(axis=0)

mm = X_max - X_min
mm[mm == 0] = 1.0

X_train_normalized = (X_train_clipped - X_min) / mm
X_validation_normalized = (X_validation_clipped - X_min) / mm
X_test_normalized = (X_test_clipped - X_min) / mm

print("\n" + "=" * 60)
print("4. ПРОВЕРКА MIN-MAX НОРМАЛИЗАЦИИ")
print("=" * 60)

norm_stats = pd.DataFrame({
    "Train_Min": X_train_normalized.min(axis=0).round(4),
    "Train_Max": X_train_normalized.max(axis=0).round(4),
    "Val_Min": X_validation_normalized.min(axis=0).round(4),
    "Val_Max": X_validation_normalized.max(axis=0).round(4),
    "Test_Min": X_test_normalized.min(axis=0).round(4),
    "Test_Max": X_test_normalized.max(axis=0).round(4),
})
print(norm_stats.to_string())

assert np.allclose(
    X_train_normalized.min(), 0
), "Минимум на Train после нормализации не равен 0!"
assert np.allclose(
    X_train_normalized.max(), 1
), "Максимум на Train после нормализации не равен 1!"

for name, arr in [
    ("Train", X_train_normalized),
    ("Validation", X_validation_normalized),
    ("Test", X_test_normalized),
]:
    assert not arr.isna().any().any(), f"Обнаружены NaN в {name} после нормализации!"

print("\n[OK] Все проверки пройдены успешно, утечек данных и NaN не обнаружено.")

# 1. Распределение классов (объединяем количество и проценты в одну аккуратную таблицу)
print("=" * 55)
print("1. РАСПРЕДЕЛЕНИЕ КЛАССОВ В ВЫБОРКАХ")
print("=" * 55)

for name, target in [
    ("Train (Обучающая)", y_train),
    ("Validation (Валидационная)", y_validation),
    ("Test (Тестовая)", y_test)
]:
    summary_df = pd.DataFrame({
        "Количество (шт.)": target.value_counts(),
        "Доля (%)": (target.value_counts(normalize=True) * 100).round(2)
    })
    print(f"\nВыборка: {name} | Всего строк: {len(target)}")
    print(summary_df)


# 2. Метрики базовой модели
print("\n" + "=" * 55)
print("2. BASELINE МОДЕЛЬ (константное предсказание: все нули)")
print("=" * 55)

baseline_predictions = np.zeros(len(y_test), dtype=int)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)

print(f"Точность (Accuracy): {baseline_accuracy:.4f} ({baseline_accuracy * 100:.2f}%)\n")
print("Подробный отчет по классам (Classification Report):")
print(classification_report(
    y_test,
    baseline_predictions,
    zero_division=0
))


# 3. Корреляция признаков с целевой переменной
print("=" * 55)
print("3. КОРРЕЛЯЦИЯ ПРИЗНАКОВ С TARGET (на Train выборке)")
print("=" * 55)

# Добавлено numeric_only=True на случай, если в X_train остались категориальные/текстовые колонки
corr_target = pd.concat(
    [X_train, y_train.rename("target")],
    axis=1
).corr(numeric_only=True)["target"].drop("target")

print("Отсортировано от сильной прямой до сильной обратной связи:\n")
print(corr_target.sort_values(ascending=False).round(4))
print("=" * 55)

# mpl.plot_correlation_matrix(X)
# mpl.plot_price_splits(data["close"], X_train, X_validation, X_test, window = 24)

#все отладочные принты описывал gemini

model = ai.LogisticRegression(learning_rate=1, epochs=10000)
model.fit(X_train, y_train)
y_hat = model.predict(X_validation)

# mpl.plot_match_over_time(y_validation, y_hat)