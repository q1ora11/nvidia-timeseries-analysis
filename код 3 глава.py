"""
Анализ временного ряда: NVIDIA Daily Stock Prices
Дисциплина: Введение в проектную деятельность
Задача: Прогнозирование стоимости акций
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Этап 1. Загрузка и первичное знакомство с данными
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("ЭТАП 1. ЗАГРУЗКА И ПЕРВИЧНОЕ ЗНАКОМСТВО С ДАННЫМИ")
print("=" * 60)

df = pd.read_csv("dataset/NVDA_yfinance_clean.csv")

print("\nПервые 5 строк:")
print(df.head())

print("\nОбщая информация:")
print(df.info())

print(f"\nРазмерность датасета: {df.shape[0]} строк × {df.shape[1]} столбцов")

# Преобразование даты и установка индекса
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)
df.set_index("Date", inplace=True)

# Числовые каналы (все кроме Volume — разные единицы)
channels = ["Close", "High", "Low", "Open", "Volume"]

print(f"\nПериод данных: {df.index.min().date()} — {df.index.max().date()}")
print(f"Количество торговых дней: {len(df)}")
print(f"Каналы: {list(df.columns)}")
print(f"Типы данных:\n{df.dtypes}")

# ─────────────────────────────────────────────────────────────────────────────
# Этап 2. Визуализация исходных данных
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ЭТАП 2. ВИЗУАЛИЗАЦИЯ ИСХОДНЫХ ДАННЫХ")
print("=" * 60)

# Граница обучающей/тестовой выборки: 80/20
split_idx = int(len(df) * 0.8)
split_date = df.index[split_idx]

price_channels = ["Close", "High", "Low", "Open"]
fig, axes = plt.subplots(len(channels), 1, figsize=(14, 14), sharex=False)
fig.suptitle("Этап 2. Визуализация исходных данных\nNVIDIA Stock (2016–2026)", fontsize=14, fontweight="bold")

for i, col in enumerate(channels):
    ax = axes[i]
    train = df[col][:split_date]
    test = df[col][split_date:]
    ax.plot(train.index, train.values, color="#1f77b4", linewidth=0.9, label="Обучающая выборка")
    ax.plot(test.index, test.values, color="#ff7f0e", linewidth=0.9, label="Тестовая выборка")
    ax.axvline(split_date, color="red", linestyle="--", linewidth=1.2, label="Граница разбивки")
    ax.set_ylabel(col, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"Канал: {col}", fontsize=9)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/stage2_visualization.png", dpi=150, bbox_inches="tight")
plt.close()
print("График сохранён: stage2_visualization.png")

# ─────────────────────────────────────────────────────────────────────────────
# Этап 3. Статистический анализ
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ЭТАП 3. СТАТИСТИЧЕСКИЙ АНАЛИЗ")
print("=" * 60)

stats = df.describe().T
stats.columns = ["count", "mean", "std", "min", "Q1", "median", "Q3", "max"]
print("\nОписательные статистики:")
print(stats.round(4).to_string())

# Частота дискретизации
time_diffs = df.index.to_series().diff().dropna()
freq_days = time_diffs.dt.days.value_counts()
print(f"\nЧастота дискретизации (торговые дни):")
print(freq_days.head())
print(f"Медианный шаг: {time_diffs.dt.days.median()} дня")

# ─────────────────────────────────────────────────────────────────────────────
# Этап 4. Анализ пропусков и выбросов
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ЭТАП 4. АНАЛИЗ ПРОПУСКОВ И ВЫБРОСОВ")
print("=" * 60)

# Пропущенные значения
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print("\nПропущенные значения:")
for col in df.columns:
    print(f"  {col}: {missing[col]} ({missing_pct[col]}%)")

# Выбросы по правилу 3 сигм
print("\nВыбросы по правилу трёх сигм:")
outlier_counts = {}
for col in channels:
    mean = df[col].mean()
    std = df[col].std()
    mask = (df[col] < mean - 3 * std) | (df[col] > mean + 3 * std)
    outlier_counts[col] = mask.sum()
    print(f"  {col}: {mask.sum()} выбросов ({mask.sum()/len(df)*100:.2f}%)")

# Диаграммы размаха для каждого канала
fig, axes = plt.subplots(1, len(channels), figsize=(16, 5))
fig.suptitle("Этап 4. Диаграммы размаха по каналам", fontsize=13, fontweight="bold")
for i, col in enumerate(channels):
    axes[i].boxplot(df[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor="#a8d8ea", color="#1f77b4"),
                    medianprops=dict(color="red", linewidth=2))
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/stage4_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("График сохранён: stage4_boxplots.png")

# ─────────────────────────────────────────────────────────────────────────────
# Этап 5. Анализ диапазонов значений
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ЭТАП 5. АНАЛИЗ ДИАПАЗОНОВ ЗНАЧЕНИЙ")
print("=" * 60)

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("Этап 5. Сравнение диапазонов значений по каналам", fontsize=13, fontweight="bold")
data_to_plot = [df[col].dropna().values for col in channels]
bp = ax.boxplot(data_to_plot, labels=channels, patch_artist=True,
                boxprops=dict(facecolor="#ffd6a5", color="#e07b39"),
                medianprops=dict(color="red", linewidth=2))
ax.set_ylabel("Значение")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/stage5_ranges.png", dpi=150, bbox_inches="tight")
plt.close()
print("График сохранён: stage5_ranges.png")

for col in channels:
    rng = df[col].max() - df[col].min()
    print(f"  {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, диапазон={rng:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Этап 6. Корреляционный анализ
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ЭТАП 6. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("=" * 60)

corr_matrix = df[channels].corr(method="pearson")
print("\nМатрица корреляции Пирсона:")
print(corr_matrix.round(4).to_string())

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle("Этап 6. Тепловая карта корреляций", fontsize=13, fontweight="bold")
im = ax.imshow(corr_matrix.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Коэффициент Пирсона")
ax.set_xticks(range(len(channels)))
ax.set_yticks(range(len(channels)))
ax.set_xticklabels(channels, rotation=45, ha="right")
ax.set_yticklabels(channels)
for i in range(len(channels)):
    for j in range(len(channels)):
        ax.text(j, i, f"{corr_matrix.values[i, j]:.2f}",
                ha="center", va="center", fontsize=10,
                color="black" if abs(corr_matrix.values[i, j]) < 0.8 else "white")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/stage6_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("График сохранён: stage6_correlation.png")

# Топ корреляций с Close
print("\nКорреляция каналов с Close (целевая переменная):")
print(corr_matrix["Close"].drop("Close").sort_values(ascending=False).round(4))

# ─────────────────────────────────────────────────────────────────────────────
# Этап 7. Поиск и анализ шумов
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ЭТАП 7. АНАЛИЗ ШУМА И ДЕКОМПОЗИЦИЯ")
print("=" * 60)

# Для биржевых данных используем канал Close
# Нет классической суточной сезонности — используем недельную (5 торговых дней)
target = df["Close"].dropna()
period = 5  # недельная сезонность (5 торговых дней)

decomp = seasonal_decompose(target, model="additive", period=period)

trend = decomp.trend.dropna()
seasonal = decomp.seasonal
residual = decomp.resid.dropna()

# Для SNR выравниваем по общему индексу
common_idx = trend.index.intersection(residual.index)
signal = (trend.loc[common_idx] + seasonal.loc[common_idx])
noise = residual.loc[common_idx]

var_signal = np.var(signal)
var_noise = np.var(noise)
snr = 10 * np.log10(var_signal / var_noise)
print(f"\nSNR (signal-to-noise ratio): {snr:.2f} дБ")

# Визуализация декомпозиции
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle("Этап 7. Декомпозиция временного ряда (Close)\n"
             f"Аддитивная модель, период={period} (торговая неделя)", fontsize=13, fontweight="bold")

axes[0].plot(target.index, target.values, color="#1f77b4", linewidth=0.8)
axes[0].set_title("Исходный ряд (Close)", fontsize=10)
axes[0].set_ylabel("Цена")
axes[0].grid(True, alpha=0.3)

axes[1].plot(decomp.trend.index, decomp.trend.values, color="#2ca02c", linewidth=1.2)
axes[1].set_title("Тренд", fontsize=10)
axes[1].set_ylabel("Тренд")
axes[1].grid(True, alpha=0.3)

axes[2].plot(decomp.seasonal.index, decomp.seasonal.values, color="#ff7f0e", linewidth=0.8)
axes[2].set_title("Сезонная компонента", fontsize=10)
axes[2].set_ylabel("Сезонность")
axes[2].grid(True, alpha=0.3)

axes[3].plot(decomp.resid.index, decomp.resid.values, color="#d62728", linewidth=0.7, alpha=0.8)
axes[3].axhline(0, color="black", linestyle="--", linewidth=0.8)
axes[3].set_title(f"Остатки (шум) | SNR = {snr:.2f} дБ", fontsize=10)
axes[3].set_ylabel("Остатки")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/stage7_decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
print("График сохранён: stage7_decomposition.png")

# Гистограмма остатков
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Этап 7. Распределение остатков (шума)", fontsize=13, fontweight="bold")
ax.hist(residual.dropna().values, bins=50, color="#9467bd", edgecolor="white", alpha=0.85)
ax.axvline(residual.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Среднее = {residual.mean():.4f}")
ax.set_xlabel("Значение остатков")
ax.set_ylabel("Частота")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/stage7_residuals_hist.png", dpi=150, bbox_inches="tight")
plt.close()
print("График сохранён: stage7_residuals_hist.png")

print(f"\nДисперсия сигнала: {var_signal:.6f}")
print(f"Дисперсия шума:    {var_noise:.6f}")
print(f"SNR:               {snr:.2f} дБ")
skewness = pd.Series(residual).skew()
kurtosis = pd.Series(residual).kurtosis()
print(f"Асимметрия остатков: {skewness:.4f}")
print(f"Эксцесс остатков:    {kurtosis:.4f}")

print("\n" + "=" * 60)
print("АНАЛИЗ ЗАВЕРШЁН. Все графики сохранены в outputs/")
print("=" * 60)
