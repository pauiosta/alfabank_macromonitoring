import numpy as np
import pandas as pd

# --- 1) Подготовка ---
df = df.copy()

# даты
df["UTILIZATION_DTTM"] = pd.to_datetime(df["UTILIZATION_DTTM"], errors="coerce", utc=True)

# список метрик-флагов
flag_cols = ["FPD_7PLUS_FLG", "D2P3_FLG", "D4P6_FLG", "D4P9_FLG", "D4P12_FLG"]

# привести флаги к 0/1/NaN
for c in flag_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")  # сохраняет NaN
    # если вдруг есть True/False:
    df[c] = df[c].replace({True: 1, False: 0})

# --- 2) Функции для доверительных интервалов биномиальным методом ---

def wilson_ci(k: int, n: int, alpha: float = 0.05):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    # z для 95% по умолчанию. Для другого alpha можно через scipy, но без scipy сделаем словарь:
    # Если нужен произвольный alpha — скажи, добавлю scipy.stats.norm.ppf
    z_map = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.01: 2.5758293035489004}
    z = z_map.get(alpha, 1.959963984540054)

    phat = k / n
    denom = 1 + (z**2) / n
    center = (phat + (z**2) / (2*n)) / denom
    half = (z * np.sqrt((phat*(1-phat) + (z**2)/(4*n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)

def agg_flag(g: pd.Series, alpha: float = 0.05):
    """Считает base, events, rate, CI для одного флага (Series с 0/1/NaN)."""
    base = int(g.notna().sum())
    events = int(g.fillna(0).sum())  # NaN не считаем событиями
    rate = events / base if base > 0 else np.nan
    lo, hi = wilson_ci(events, base, alpha=alpha)
    return pd.Series({
        "base": base,
        "events": events,
        "rate": rate,
        "ci_low": lo,
        "ci_high": hi
    })

# --- 3) Сводная таблица по тестам ---
alpha = 0.05  # 95% CI

group_cols = ["TEST_NAME", "TEST_GROUP_NAME"]

# базовые агрегаты
summary = (
    df.groupby(group_cols, dropna=False)
      .agg(
          n_obs=("APPLICATION_RK", "count"),  # или ("OFFER_RK","count") - что считаешь наблюдением
          min_utilization_dttm=("UTILIZATION_DTTM", "min"),
          max_utilization_dttm=("UTILIZATION_DTTM", "max"),
      )
      .reset_index()
)

# метрики (для каждой добавляем base/events/rate/ci)
for c in flag_cols:
    m = (
        df.groupby(group_cols, dropna=False)[c]
          .apply(lambda s: agg_flag(s, alpha=alpha))
          .reset_index()
    )
    # колонкам дадим префикс метрики
    m = m.rename(columns={
        "base": f"{c}__base",
        "events": f"{c}__events",
        "rate": f"{c}__rate",
        "ci_low": f"{c}__ci_low",
        "ci_high": f"{c}__ci_high",
    })
    summary = summary.merge(m, on=group_cols, how="left")

summary


group_cols = ["TEST_NAME", "TEST_GROUP_NAME"]
flag_cols = ["FPD_7PLUS_FLG", "D2P3_FLG", "D4P6_FLG", "D4P9_FLG", "D4P12_FLG"]

def agg_flag(s, alpha=0.05):
    base = int(s.notna().sum())
    events = int(s.fillna(0).sum())
    rate = events / base if base else np.nan
    lo, hi = wilson_ci(events, base, alpha=alpha)
    return pd.Series({"base": base, "events": events, "rate": rate, "ci_low": lo, "ci_high": hi})

summary = (
    df.groupby(group_cols, dropna=False)
      .agg(
          n_obs=("APPLICATION_RK", "count"),
          min_utilization_dttm=("UTILIZATION_DTTM", "min"),
          max_utilization_dttm=("UTILIZATION_DTTM", "max"),
      )
)

# считаем метрики и “расплющиваем” в wide-формат
metrics = []
for c in flag_cols:
    m = df.groupby(group_cols, dropna=False)[c].apply(agg_flag)
    # m имеет MultiIndex (group_cols + metric_name). unstack превращает metric_name в колонки
    m = m.unstack()  # columns: base/events/rate/ci_low/ci_high
    m = m.add_prefix(f"{c}__")  # FPD_7PLUS_FLG__base и т.д.
    metrics.append(m)

summary = summary.join(metrics, how="left").reset_index()
summary
