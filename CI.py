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


summary["min_utilization_dttm"] = pd.to_datetime(summary["min_utilization_dttm"]).dt.date
summary["max_utilization_dttm"] = pd.to_datetime(summary["max_utilization_dttm"]).dt.date

summary.to_excel("xsell_summary.xlsx", index=False)



import numpy as np
import pandas as pd

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)

    z_map = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.01: 2.5758293035489004}
    z = z_map.get(alpha, 1.959963984540054)

    phat = k / n
    denom = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2 * n)) / denom
    half = (z * np.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)) / denom

    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


def metric_ci_for_group(
    df: pd.DataFrame,
    test_name: str,
    test_group_name: str,
    metric_col: str,               # "FPD_7PLUS_FLG" / "D2P3_FLG" / "D4P6_FLG" ...
    cuts: list[str] | None = None, # доп разрезы, напр ["TERM"] или ["TERM","MONTHLY_CREDIT_RATE"]
    alpha: float = 0.05,
    date_format: str = "%Y-%m-%d",
    col_test: str = "TEST_NAME",
    col_group: str = "TEST_GROUP_NAME",
    col_util_dt: str = "UTILIZATION_DTTM",
) -> pd.DataFrame:
    """
    Считает метрики по одному test_name + test_group_name + (опционально) разрезам cuts.
    Возвращает таблицу: по каждой группе -> n_obs, min/max util_date, base/events/rate + Wilson CI.
    base = count(notna(flag)), events = sum(flag, NaN=0).
    """
    cuts = cuts or []

    # проверки колонок
    need_cols = [col_test, col_group, col_util_dt, metric_col] + cuts
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in df: {missing}")

    # фильтр по тесту/группе
    d = df[(df[col_test] == test_name) & (df[col_group] == test_group_name)].copy()

    # util date -> datetime
    d[col_util_dt] = pd.to_datetime(d[col_util_dt], errors="coerce")

    group_cols = [col_test, col_group] + cuts

    def _agg_one(g: pd.DataFrame) -> pd.Series:
        util = g[col_util_dt]
        flag = g[metric_col]

        n_obs = int(len(g))
        min_dt = util.min()
        max_dt = util.max()

        base = int(flag.notna().sum())
        events = int(flag.fillna(0).sum())
        rate = events / base if base > 0 else np.nan
        ci_low, ci_high = wilson_ci(events, base, alpha=alpha)

        return pd.Series({
            "n_obs": n_obs,
            "min_utilization_dttm": min_dt.strftime(date_format) if pd.notna(min_dt) else None,
            "max_utilization_dttm": max_dt.strftime(date_format) if pd.notna(max_dt) else None,
            "base": base,
            "events": events,
            "rate": rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })

    out = (
        d.groupby(group_cols, dropna=False)
         .apply(_agg_one)
         .reset_index()
    )

    # чтобы было видно какой metric считали
    out.insert(out.columns.get_loc("n_obs"), "METRIC", metric_col)

    return out


# ===== примеры =====
# 1) без разрезов (как раньше)
# res = metric_ci_for_group(df, "RBP_CONT_MIDTERM", "CHALLENGER_AMT", "D4P6_FLG")

# 2) разрез по term
# res = metric_ci_for_group(df, "RBP_CONT_MIDTERM", "CHALLENGER_AMT", "D4P6_FLG", cuts=["TERM"])

# 3) разрез по term + rate
# res = metric_ci_for_group(df, "RBP_CONT_MIDTERM", "CHALLENGER_AMT", "D4P6_FLG",
#                           cuts=["TERM", "MONTHLY_CREDIT_RATE"])


import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import norm, ttest_ind

# ---------------------------
# Helpers: Wilson CI for proportion
# ---------------------------
def wilson_ci(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - alpha/2)
    phat = k / n
    denom = 1 + (z**2)/n
    center = (phat + (z**2)/(2*n)) / denom
    half = (z * sqrt((phat*(1-phat) + (z**2)/(4*n))/n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi

# ---------------------------
# MDE for proportions (approx, two-sided)
# ---------------------------
def mde_prop(p: float, n1: int, n2: int, alpha: float = 0.05, power: float = 0.8):
    if n1 == 0 or n2 == 0 or np.isnan(p):
        return np.nan
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta  = norm.ppf(power)
    se = sqrt(p*(1-p)*(1/n1 + 1/n2))
    return (z_alpha + z_beta) * se  # absolute MDE in probability units

# ---------------------------
# MDE for means (approx, pooled SD)
# ---------------------------
def mde_mean(sd_pooled: float, n1: int, n2: int, alpha: float = 0.05, power: float = 0.8):
    if n1 == 0 or n2 == 0 or np.isnan(sd_pooled):
        return np.nan
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta  = norm.ppf(power)
    se = sd_pooled * sqrt(1/n1 + 1/n2)
    return (z_alpha + z_beta) * se  # absolute MDE in units of the metric

# ---------------------------
# Main compare
# metrics_spec example:
# [
#   {"name":"conv_offer_to_utilization_2m", "type":"prop"},  # 0/1 or boolean or already rate by row
#   {"name":"prod_disbursement_amt", "type":"mean"},
#   ...
# ]
# ---------------------------
def compare_two_groups(
    df: pd.DataFrame,
    test_name: str,
    group_a: str,
    group_b: str,
    metrics_spec: list,
    cuts: list | None = None,
    alpha: float = 0.05,
    power: float = 0.8
) -> pd.DataFrame:

    cuts = cuts or []
    base_cols = ["TEST_NAME", "TEST_GROUP_NAME"] + cuts

    d = df[df["TEST_NAME"] == test_name].copy()
    d = d[d["TEST_GROUP_NAME"].isin([group_a, group_b])].copy()

    out_rows = []

    # будем считать отдельно по каждому разрезу (если cuts пустой — будет один блок)
    if cuts:
        keys = d[cuts].drop_duplicates()
        keys_iter = keys.to_dict("records")
    else:
        keys_iter = [dict()]

    for key in keys_iter:
        dd = d.copy()
        for c, v in key.items():
            dd = dd[dd[c] == v]

        for m in metrics_spec:
            col = m["name"]
            mtype = m.get("type", "mean")  # mean/prop

            a = dd[dd["TEST_GROUP_NAME"] == group_a][col]
            b = dd[dd["TEST_GROUP_NAME"] == group_b][col]

            # drop NaN
            a_non = a.dropna()
            b_non = b.dropna()

            nA = len(a_non)
            nB = len(b_non)

            # значения по группам
            if mtype == "prop":
                # считаем events как сумму 1-иц
                kA = int(a_non.fillna(0).sum())
                kB = int(b_non.fillna(0).sum())
                pA = kA / nA if nA > 0 else np.nan
                pB = kB / nB if nB > 0 else np.nan

                # значимость (z-test разности долей, pooled)
                if nA > 0 and nB > 0:
                    p_pool = (kA + kB) / (nA + nB) if (nA + nB) > 0 else np.nan
                    se = sqrt(p_pool*(1-p_pool)*(1/nA + 1/nB)) if p_pool == p_pool else np.nan
                    z = (pB - pA)/se if se and se > 0 else np.nan
                    pval = 2*(1 - norm.cdf(abs(z))) if z == z else np.nan
                else:
                    pval = np.nan

                uplift = ((pB - pA) / pA) if (pA and pA != 0 and pA == pA and pB == pB) else np.nan
                mde = mde_prop(pA, nA, nB, alpha=alpha, power=power)  # абсолютный MDE в долях

                valA, valB = pA, pB

            else:
                # mean metric
                meanA = float(a_non.mean()) if nA > 0 else np.nan
                meanB = float(b_non.mean()) if nB > 0 else np.nan

                # значимость (Welch t-test)
                if nA > 1 and nB > 1:
                    stat, pval = ttest_ind(a_non, b_non, equal_var=False, nan_policy="omit")
                else:
                    pval = np.nan

                uplift = ((meanB - meanA) / meanA) if (meanA and meanA != 0 and meanA == meanA and meanB == meanB) else np.nan

                # pooled sd для MDE (приближение; можно оставлять Welch, но MDE обычно по pooled)
                if nA > 1 and nB > 1:
                    sdA = float(a_non.std(ddof=1))
                    sdB = float(b_non.std(ddof=1))
                    sd_pooled = sqrt(((nA-1)*sdA**2 + (nB-1)*sdB**2) / (nA + nB - 2))
                else:
                    sd_pooled = np.nan

                mde = mde_mean(sd_pooled, nA, nB, alpha=alpha, power=power)  # абсолютный MDE

                valA, valB = meanA, meanB

            significant = "YES" if (pval == pval and pval < alpha) else "NO"

            row = {
                **key,
                "Metric name": col,
                group_a: valA,
                group_b: valB,
                "Significant?": significant,
                "p_value": pval,
                "Uplift": uplift,
                "MDE": mde,
                "n_A": nA,
                "n_B": nB,
            }
            out_rows.append(row)

    res = pd.DataFrame(out_rows)

    # красивый формат uplift как %
    if "Uplift" in res.columns:
        res["Uplift"] = res["Uplift"].apply(lambda x: np.nan if pd.isna(x) else x)

    return res
    
    
    metrics = [
    {"name": "conv_offer_to_utilization_2m", "type": "prop"},
    {"name": "D2P3_FLG", "type": "prop"},
    {"name": "FPD_7PLUS_FLG", "type": "prop"},
    {"name": "pd", "type": "mean"},
    {"name": "prod_disbursement_amt", "type": "mean"},
    {"name": "prod_loan_term_month", "type": "mean"},
    {"name": "prod_monthly_credit_rate", "type": "mean"},
    {"name": "prod_origination_fee_rate", "type": "mean"},
    {"name": "prod_regular_payment_amt", "type": "mean"},
    {"name": "prod_vas_flg", "type": "prop"},
]

tbl = compare_two_groups(
    df,
    test_name="RBP_CONT_MIDTERM",
    group_a="BASIC",
    group_b="CHALLENGER_AMT",
    metrics_spec=metrics,
    cuts=None,          # без разрезов
    alpha=0.05,
    power=0.8
)

tbl



tbl_term = compare_two_groups(
    df,
    test_name="RBP_BAD",
    group_a="bad_basic",
    group_b="bad_bad",
    metrics_spec=[{"name":"D4P12_FLG", "type":"prop"}],
    cuts=["REQUESTED_TERM", "MAXLOANAMTGROUP"]
)

# оставить только term=6 и maxloanamtgroup=10k
tbl_term = tbl_term[(tbl_term["REQUESTED_TERM"] == 6) & (tbl_term["MAXLOANAMTGROUP"] == "10k")]
tbl_term




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_two_distributions(
    df: pd.DataFrame,
    test_name: str,
    group_a: str,
    group_b: str,
    col: str,
    bins: int = 30,
    normalize: str = "density",   # "density" or "count"
    dropna: bool = True,
    title: str | None = None,
):
    """
    One chart: two distributions (A vs B) inside one test for a chosen column.

    normalize:
      - "density" -> probability density (areas ~ 1)
      - "count"   -> raw counts
    """
    d = df[df["TEST_NAME"].eq(test_name)].copy()

    if dropna:
        d = d[d[col].notna()]

    a = d[d["TEST_GROUP_NAME"].eq(group_a)][col]
    b = d[d["TEST_GROUP_NAME"].eq(group_b)][col]

    # Convert to numeric if possible (safe for ints/floats stored as strings)
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    a = a.dropna()
    b = b.dropna()

    if a.empty or b.empty:
        raise ValueError(
            f"No data after filtering. Sizes: A={len(a)}, B={len(b)}. "
            f"Check test_name/group names/column."
        )

    # Shared bin edges so histograms are comparable
    data_min = np.nanmin([a.min(), b.min()])
    data_max = np.nanmax([a.max(), b.max()])
    if data_min == data_max:
        data_min -= 0.5
        data_max += 0.5
    bin_edges = np.linspace(data_min, data_max, bins + 1)

    density = (normalize == "density")
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(a, bins=bin_edges, alpha=0.45, density=density, label=f"{group_a} (n={len(a)})")
    ax.hist(b, bins=bin_edges, alpha=0.45, density=density, label=f"{group_b} (n={len(b)})")

    ax.set_xlabel(col)
    ax.set_ylabel("Density" if density else "Count")
    ax.legend()

    if title is None:
        title = f"{test_name}: {col} distribution — {group_a} vs {group_b}"
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


# ---- Example usage ----
# plot_two_distributions(
#     df,
#     test_name="RBP_GOOD",
#     group_a="good_basic",
#     group_b="good_good",
#     col="REQUESTED_TERM",
#     bins=20,
#     normalize="density",
# )


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def wilson_ci(k: int, n: int, alpha: float = 0.05):
    """Wilson score CI for binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    # z for common alphas (two-sided)
    z_map = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.01: 2.5758293035489004}
    z = z_map.get(alpha, 1.959963984540054)

    phat = k / n
    denom = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2*n)) / denom
    half = (z * np.sqrt((phat*(1-phat) + (z**2)/(4*n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def plot_risk_by_term_two_groups(
    df: pd.DataFrame,
    test_name: str,
    group_a: str,
    group_b: str,
    metric_col: str,                 # e.g. "D4P12_FLG" / "D4P6_FLG" / "FPD_7PLUS_FLG"
    term_col: str = "REQUESTED_TERM",
    alpha: float = 0.05,
    term_filter: list[int] | None = None,   # e.g. [6, 9, 12, 18]
    extra_filters: dict | None = None,      # e.g. {"MAXLOANAMTGROUP": "10k"}
    title: str | None = None,
):
    """
    Risk (rate) by requested term for two groups within one test.
    Computes base/events/rate + Wilson CI and plots rate with errorbars.
    """
    d = df[df["TEST_NAME"].eq(test_name)].copy()

    # optional extra filters
    if extra_filters:
        for c, v in extra_filters.items():
            d = d[d[c].eq(v)]

    # term filter
    if term_filter is not None:
        d = d[d[term_col].isin(term_filter)]

    # keep only needed groups
    d = d[d["TEST_GROUP_NAME"].isin([group_a, group_b])].copy()

    # ensure term numeric (safe)
    d[term_col] = pd.to_numeric(d[term_col], errors="coerce")
    d = d[d[term_col].notna()].copy()

    # metric to numeric (0/1); NaN means "not matured" and excluded from base
    m = pd.to_numeric(d[metric_col], errors="coerce")
    d["_m"] = m

    def agg_group(g: pd.DataFrame):
        base = int(g["_m"].notna().sum())
        events = int(g["_m"].fillna(0).sum())  # assumes flags are 0/1
        rate = events / base if base > 0 else np.nan
        lo, hi = wilson_ci(events, base, alpha=alpha)
        return pd.Series({"base": base, "events": events, "rate": rate, "ci_low": lo, "ci_high": hi})

    out = (
        d.groupby(["TEST_GROUP_NAME", term_col], dropna=False)
         .apply(agg_group)
         .reset_index()
         .sort_values([term_col, "TEST_GROUP_NAME"])
    )

    # Split for plotting
    a = out[out["TEST_GROUP_NAME"].eq(group_a)].sort_values(term_col)
    b = out[out["TEST_GROUP_NAME"].eq(group_b)].sort_values(term_col)

    if a.empty or b.empty:
        raise ValueError(
            f"No data to plot. After filters, rows: A={len(a)}, B={len(b)}. "
            f"Check test/group/metric/filters."
        )

    fig, ax = plt.subplots(figsize=(10, 5))

    # error bars are asymmetric
    ax.errorbar(
        a[term_col], a["rate"],
        yerr=[a["rate"] - a["ci_low"], a["ci_high"] - a["rate"]],
        fmt="o-", capsize=3, label=f"{group_a}"
    )
    ax.errorbar(
        b[term_col], b["rate"],
        yerr=[b["rate"] - b["ci_low"], b["ci_high"] - b["rate"]],
        fmt="o-", capsize=3, label=f"{group_b}"
    )

    ax.set_xlabel(term_col)
    ax.set_ylabel(f"Risk rate ({metric_col})")
    ax.set_xticks(sorted(pd.unique(out[term_col].dropna()).tolist()))
    ax.legend()

    if title is None:
        title = f"{test_name}: {metric_col} risk by {term_col} — {group_a} vs {group_b}"
        if extra_filters:
            title += " | " + ", ".join([f"{k}={v}" for k, v in extra_filters.items()])
    ax.set_title(title)

    plt.tight_layout()
    plt.show()

    return out


# ---- Example usage ----
# res = plot_risk_by_term_two_groups(
#     df,
#     test_name="RBP_GOOD",
#     group_a="good_basic",
#     group_b="good_good",
#     metric_col="D4P12_FLG",
#     term_col="REQUESTED_TERM",
#     alpha=0.05,
#     term_filter=[6,9,12,15,18],
#     extra_filters={"MAXLOANAMTGROUP": "10k"}  # optional
# )
# res

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_maxloanamtgroup_distribution(
    df: pd.DataFrame,
    test_name: str,
    group_a: str,
    group_b: str,
    col: str = "MAXLOANAMTGROUP",
    normalize: bool = True,          # True -> доли, False -> counts
    extra_filters: dict | None = None,
    title: str | None = None,
):
    d = df[df["TEST_NAME"].eq(test_name)].copy()
    d = d[d["TEST_GROUP_NAME"].isin([group_a, group_b])].copy()

    if extra_filters:
        for c, v in extra_filters.items():
            d = d[d[c].eq(v)]

    # считаем распределение
    tab = (
        d.pivot_table(index=col, columns="TEST_GROUP_NAME", values="APPLICATION_RK",
                      aggfunc="count", fill_value=0)
        .reindex(columns=[group_a, group_b])
    )

    if normalize:
        tab = tab.div(tab.sum(axis=0), axis=1)  # доли внутри каждой группы

    # сортировка категорий “по размеру” (если есть числа в строке)
    def _num(x):
        s = str(x).lower().replace("<= ", "").replace("<=", "").replace("k", "")
        try:
            return float(s)
        except:
            return np.nan

    tab = tab.loc[sorted(tab.index, key=lambda x: (np.isnan(_num(x)), _num(x), str(x)))]

    ax = tab.plot(kind="bar", figsize=(10, 5))
    ax.set_xlabel(col)
    ax.set_ylabel("Share" if normalize else "Count")
    ax.legend(title="TEST_GROUP_NAME")

    if title is None:
        title = f"{test_name}: {col} distribution — {group_a} vs {group_b}"
        if extra_filters:
            title += " | " + ", ".join([f"{k}={v}" for k, v in extra_filters.items()])
    ax.set_title(title)

    plt.tight_layout()
    plt.show()

    return tab


# пример:
# dist = plot_maxloanamtgroup_distribution(
#     df,
#     test_name="RBP_GOOD",
#     group_a="good_basic",
#     group_b="good_good",
#     col="MAXLOANAMTGROUP",
#     normalize=True
# )
# 



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (np.nan, np.nan)
    z_map = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.01: 2.5758293035489004}
    z = z_map.get(alpha, 1.959963984540054)

    phat = k / n
    denom = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2*n)) / denom
    half = (z * np.sqrt((phat*(1-phat) + (z**2)/(4*n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def plot_risk_by_amt_two_groups(
    df: pd.DataFrame,
    test_name: str,
    group_a: str,
    group_b: str,
    metric_col: str,                      # e.g. "D4P12_FLG"
    amt_col: str = "MAXLOANAMTGROUP",     # bucket column like "10k", "25k", etc.
    alpha: float = 0.05,
    amt_order: list[str] | None = None,  # optional explicit order: ["10k","25k","50k"]
    extra_filters: dict | None = None,   # e.g. {"REQUESTED_TERM": 6}
    title: str | None = None,
):
    """
    Risk (rate) by amount bucket (MAXLOANAMTGROUP) for two groups within one test.
    Computes base/events/rate + Wilson CI and plots rate with errorbars.
    """
    d = df[df["TEST_NAME"].eq(test_name)].copy()

    if extra_filters:
        for c, v in extra_filters.items():
            d = d[d[c].eq(v)]

    d = d[d["TEST_GROUP_NAME"].isin([group_a, group_b])].copy()

    # metric: NaN means not matured => excluded from base
    d["_m"] = pd.to_numeric(d[metric_col], errors="coerce")

    # keep amount bucket as string/categorical
    d[amt_col] = d[amt_col].astype(str)
    d = d[d[amt_col].notna()].copy()

    def agg_group(g: pd.DataFrame) -> pd.Series:
        base = int(g["_m"].notna().sum())
        events = int(g["_m"].fillna(0).sum())
        rate = events / base if base > 0 else np.nan
        lo, hi = wilson_ci(events, base, alpha=alpha)
        return pd.Series({"base": base, "events": events, "rate": rate, "ci_low": lo, "ci_high": hi})

    out = (
        d.groupby(["TEST_GROUP_NAME", amt_col], dropna=False)
         .apply(agg_group)
         .reset_index()
    )

    # Ordering of buckets
    if amt_order is None:
        # Try numeric sorting from strings like "10k", "25k", "10000", "10,000"
        def to_num(x: str):
            s = str(x).lower().replace(",", "").replace(" ", "")
            s = s.replace("php", "").replace("₱", "")
            if s.endswith("k"):
                try:
                    return float(s[:-1]) * 1000
                except:
                    return np.nan
            try:
                return float(s)
            except:
                return np.nan

        uniq = out[amt_col].dropna().unique().tolist()
        amt_order = sorted(uniq, key=lambda x: (np.isnan(to_num(x)), to_num(x), str(x)))

    # make categorical to control plot order
    out[amt_col] = pd.Categorical(out[amt_col], categories=amt_order, ordered=True)
    out = out.sort_values([amt_col, "TEST_GROUP_NAME"])

    a = out[out["TEST_GROUP_NAME"].eq(group_a)].sort_values(amt_col)
    b = out[out["TEST_GROUP_NAME"].eq(group_b)].sort_values(amt_col)

    if a.empty or b.empty:
        raise ValueError(
            f"No data to plot. After filters, rows: A={len(a)}, B={len(b)}. "
            f"Check test/group/metric/filters."
        )

    x = np.arange(len(amt_order))
    map_x = {cat: i for i, cat in enumerate(amt_order)}
    ax_x_a = a[amt_col].astype(str).map(map_x).to_numpy()
    ax_x_b = b[amt_col].astype(str).map(map_x).to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.errorbar(
        ax_x_a, a["rate"],
        yerr=[a["rate"] - a["ci_low"], a["ci_high"] - a["rate"]],
        fmt="o-", capsize=3, label=f"{group_a}"
    )
    ax.errorbar(
        ax_x_b, b["rate"],
        yerr=[b["rate"] - b["ci_low"], b["ci_high"] - b["rate"]],
        fmt="o-", capsize=3, label=f"{group_b}"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(amt_order, rotation=0)
    ax.set_xlabel(amt_col)
    ax.set_ylabel(f"Risk rate ({metric_col})")
    ax.legend()

    if title is None:
        title = f"{test_name}: {metric_col} risk by {amt_col} — {group_a} vs {group_b}"
        if extra_filters:
            title += " | " + ", ".join([f"{k}={v}" for k, v in extra_filters.items()])
    ax.set_title(title)

    plt.tight_layout()
    plt.show()

    return out


# ---- Example usage ----
# res_amt = plot_risk_by_amt_two_groups(
#     df,
#     test_name="RBP_GOOD",
#     group_a="good_basic",
#     group_b="good_good",
#     metric_col="D4P12_FLG",
#     amt_col="MAXLOANAMTGROUP",
#     extra_filters={"REQUESTED_TERM": 6},   # optional
#     alpha=0.05
# )
# res_amt



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def wilson_ci(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (np.nan, np.nan)
    z_map = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.01: 2.5758293035489004}
    z = z_map.get(alpha, 1.959963984540054)

    phat = k / n
    denom = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2*n)) / denom
    half = (z * np.sqrt((phat*(1-phat) + (z**2)/(4*n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def plot_risk_one_group(
    df: pd.DataFrame,
    test_name: str,
    group_name: str,
    metric_col: str,     # e.g. "D4P6_FLG", "D4P12_FLG"
    cut_col: str,        # e.g. "REQUESTED_TERM" or "MAXLOANAMTGROUP"
    alpha: float = 0.05,
    cut_values=None,     # optional list filter: [6,9,12,15,18] or ["10k","25k"]
    filters: dict | None = None,  # optional extra filters: {"REQUESTED_TERM": 6, "MAXLOANAMTGROUP": "10k"}
    title: str | None = None,
):
    d = df[df["TEST_NAME"].eq(test_name)].copy()
    d = d[d["TEST_GROUP_NAME"].eq(group_name)].copy()

    if filters:
        for c, v in filters.items():
            if c not in d.columns:
                raise ValueError(f"Filter column '{c}' not found in df.columns")
            d = d[d[c].eq(v)]

    if cut_col not in d.columns:
        raise ValueError(f"cut_col '{cut_col}' not found in df.columns")

    if cut_values is not None:
        d = d[d[cut_col].isin(cut_values)].copy()

    d["_m"] = pd.to_numeric(d[metric_col], errors="coerce")

    # агрегируем риск + CI по cut_col
    rows = []
    for val, g in d.groupby(cut_col, dropna=False):
        base = int(g["_m"].notna().sum())
        events = int(g["_m"].fillna(0).sum())
        rate = events / base if base > 0 else np.nan
        lo, hi = wilson_ci(events, base, alpha=alpha)
        rows.append((val, base, events, rate, lo, hi))

    out = pd.DataFrame(rows, columns=[cut_col, "base", "events", "rate", "ci_low", "ci_high"])

    # сортировка (если числа — по числам)
    try:
        out["_cut_num"] = pd.to_numeric(out[cut_col], errors="coerce")
        if out["_cut_num"].notna().any():
            out = out.sort_values(["_cut_num", cut_col])
        else:
            out = out.sort_values(cut_col)
        out = out.drop(columns=["_cut_num"])
    except Exception:
        out = out.sort_values(cut_col)

    if out.empty:
        raise ValueError("No data after filters (out is empty). Check test/group/metric/cut/filters.")

    # --- plot
    x = out[cut_col].astype(str).tolist()
    y = out["rate"].astype(float).values
    yerr_low = y - out["ci_low"].values
    yerr_high = out["ci_high"].values - y

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="o-", capsize=4)
    ax.set_xlabel(cut_col)
    ax.set_ylabel(f"Risk rate ({metric_col})")
    ax.set_title(title or f"{test_name} / {group_name}: {metric_col} by {cut_col} (Wilson CI)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return out
    
    
out = plot_risk_one_group(
    df,
    test_name="RBP_GOOD",
    group_name="good_good",
    metric_col="D4P12_FLG",
    cut_col="UTIL_MONTH",
    filters={"REQUESTED_TERM": 6, "MAXLOANAMTGROUP": "10k"},
)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def wilson_ci(k: int, n: int, alpha: float = 0.05):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    # z for two-sided CI
    z_map = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.01: 2.5758293035489004}
    z = z_map.get(alpha, 1.959963984540054)

    phat = k / n
    denom = 1 + (z**2) / n
    center = (phat + (z**2) / (2*n)) / denom
    half = (z * np.sqrt((phat*(1-phat) + (z**2)/(4*n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi

def _agg_binom_flag(s: pd.Series, alpha: float = 0.05):
    """
    base = кол-во НЕ NaN (наблюдения с определенным флагом)
    events = сумма флага (1/0), NaN считаем как 0 для событий, но base по ним не растим
    """
    base = int(s.notna().sum())
    events = int(s.fillna(0).sum())
    rate = events / base if base > 0 else np.nan
    lo, hi = wilson_ci(events, base, alpha=alpha)
    return pd.Series({"base": base, "events": events, "rate": rate, "ci_low": lo, "ci_high": hi})

def plot_risk_by_cut_one_group(
    df: pd.DataFrame,
    test_name: str,
    test_group_name: str,
    metric_cols=("D4P6_FLG", "D4P9_FLG", "D4P12_FLG"),
    cut_col="REQUESTED_TERM",
    cut_filter=None,                 # например [6,9,12,15,18]
    extra_filters=None,              # например {"MAXLOANAMTGROUP": "10k"}
    alpha=0.05,
    title=None
):
    extra_filters = extra_filters or {}

    d = df.loc[
        (df["TEST_NAME"] == test_name) &
        (df["TEST_GROUP_NAME"] == test_group_name)
    ].copy()

    # доп. фильтры
    for col, val in extra_filters.items():
        d = d.loc[d[col] == val]

    if cut_filter is not None:
        d = d.loc[d[cut_col].isin(cut_filter)]

    if d.empty:
        raise ValueError("No rows after filters. Check TEST_NAME/TEST_GROUP_NAME/cut/filters.")

    # агрегируем по cut_col для каждой метрики
    out_parts = []
    for m in metric_cols:
        tmp = (
            d.groupby(cut_col, dropna=False)[m]
             .apply(lambda s: _agg_binom_flag(s, alpha=alpha))
             .reset_index()
        )
        tmp["METRIC"] = m
        out_parts.append(tmp)

    out = pd.concat(out_parts, ignore_index=True)

    # сортировка по разрезу (если числовой — по числу)
    if pd.api.types.is_numeric_dtype(out[cut_col]):
        out = out.sort_values([cut_col, "METRIC"])
    else:
        out = out.sort_values([cut_col, "METRIC"])

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # x positions
    x_vals = out[cut_col].unique()
    x_map = {v: i for i, v in enumerate(x_vals)}

    for m in metric_cols:
        sub = out[out["METRIC"] == m].copy()
        xs = sub[cut_col].map(x_map).to_numpy()
        ys = sub["rate"].to_numpy()

        # asymmetric error bars
        yerr_low = ys - sub["ci_low"].to_numpy()
        yerr_high = sub["ci_high"].to_numpy() - ys
        ax.errorbar(xs, ys, yerr=[yerr_low, yerr_high], marker="o", capsize=3, label=m)

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals], rotation=0)
    ax.set_xlabel(cut_col)
    ax.set_ylabel("Risk rate")
    ax.set_title(title or f"{test_name} / {test_group_name}: risk by {cut_col}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return out
    
res = plot_risk_by_cut_one_group(
    df,
    test_name="RBP_GOOD",
    test_group_name="good_good",
    metric_cols=["D4P6_FLG", "D4P9_FLG", "D4P12_FLG"],
    cut_col="REQUESTED_TERM",
    cut_filter=[6,9,12,15,18],
    extra_filters={"MAXLOANAMTGROUP": "10k"},
)




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def wilson_ci(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (np.nan, np.nan)
    z_map = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.01: 2.5758293035489004}
    z = z_map.get(alpha, 1.959963984540054)

    phat = k / n
    denom = 1 + (z**2) / n
    center = (phat + (z**2) / (2*n)) / denom
    half = (z * np.sqrt((phat*(1-phat) + (z**2)/(4*n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi

def _agg_binom_flag(s: pd.Series, alpha: float = 0.05):
    base = int(s.notna().sum())
    events = int(s.fillna(0).sum())
    rate = events / base if base > 0 else np.nan
    lo, hi = wilson_ci(events, base, alpha=alpha)
    return pd.Series({"base": base, "events": events, "rate": rate, "ci_low": lo, "ci_high": hi})

def plot_risk_by_cut_one_group(
    df: pd.DataFrame,
    test_name: str,
    test_group_name: str,
    metric_cols=("D4P6_FLG", "D4P9_FLG", "D4P12_FLG"),
    cut_col="REQUESTED_TERM",
    cut_filter=None,
    extra_filters=None,
    alpha=0.05,
    title=None
):
    extra_filters = extra_filters or {}

    d = df.loc[
        (df["TEST_NAME"] == test_name) &
        (df["TEST_GROUP_NAME"] == test_group_name)
    ].copy()

    # доп. фильтры (важно: тут должно быть точное значение как в данных)
    for col, val in extra_filters.items():
        if col not in d.columns:
            raise KeyError(f"Column '{col}' not found in df. Available cols example: {list(d.columns)[:20]}")
        d = d.loc[d[col] == val]

    if cut_filter is not None:
        d = d.loc[d[cut_col].isin(cut_filter)]

    if d.empty:
        raise ValueError("No rows after filters. Check TEST_NAME/TEST_GROUP_NAME/cut/filters.")

    # агрегируем по cut_col для каждой метрики, делаем unstack чтобы точно получить rate/ci_low/ci_high колонками
    parts = []
    for m in metric_cols:
        if m not in d.columns:
            raise KeyError(f"Metric column '{m}' not found in df.")
        tmp = (
            d.groupby(cut_col)[m]
             .apply(lambda s: _agg_binom_flag(s, alpha=alpha))
             .unstack()               # <-- ключевой фикс против KeyError: 'rate'
             .reset_index()
        )
        tmp["METRIC"] = m
        parts.append(tmp)

    out = pd.concat(parts, ignore_index=True)

    # проверка на всякий случай
    required = {"rate", "ci_low", "ci_high"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"After aggregation missing columns: {missing}. Got columns: {list(out.columns)}")

    # сортировка
    out = out.sort_values([cut_col, "METRIC"])

    # рисуем
    fig, ax = plt.subplots(figsize=(10, 5))
    x_vals = out[cut_col].dropna().unique()
    x_vals = sorted(x_vals) if pd.api.types.is_numeric_dtype(out[cut_col]) else list(x_vals)
    x_map = {v: i for i, v in enumerate(x_vals)}

    for m in metric_cols:
        sub = out[out["METRIC"] == m].copy()
        sub = sub[sub[cut_col].isin(x_map.keys())]
        xs = sub[cut_col].map(x_map).to_numpy()
        ys = sub["rate"].to_numpy()

        yerr_low = ys - sub["ci_low"].to_numpy()
        yerr_high = sub["ci_high"].to_numpy() - ys
        ax.errorbar(xs, ys, yerr=[yerr_low, yerr_high], marker="o", capsize=3, label=m)

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals])
    ax.set_xlabel(cut_col)
    ax.set_ylabel("Risk rate")
    ax.set_title(title or f"{test_name} / {test_group_name}: risk by {cut_col}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return out
    
    
    
    
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _try_parse_number(x):
    """Extract first number from a string like '<=10K', '10k', '35K', '10000'."""
    if pd.isna(x):
        return np.nan
    s = str(x).lower().replace(",", "").strip()
    m = re.search(r"(\d+(\.\d+)?)", s)
    if not m:
        return np.nan
    val = float(m.group(1))
    # crude K handling: if contains 'k' -> thousands
    if "k" in s:
        val *= 1000
    return val

def _sort_terms(values):
    vals = pd.Series(values).dropna().unique()
    # numeric if possible
    try:
        nums = pd.to_numeric(vals)
        return sorted(nums)
    except Exception:
        # fallback string sort
        return sorted(vals, key=lambda x: str(x))

def _sort_amt_groups(values):
    vals = pd.Series(values).dropna().unique()
    # sort by extracted numeric
    return sorted(vals, key=lambda x: (_try_parse_number(x), str(x)))

def risk_heatmap_by_term_and_amt(
    df: pd.DataFrame,
    test_name: str,
    test_group_name: str,
    term_col: str = "REQUESTED_TERM",
    amt_col: str = "MAXLOANAMTGROUP",
    metrics: list = None,
    alpha: float = 0.05,  # not used here (no CI on heatmap), left for compatibility
):
    """
    Builds heatmaps: X=term_col, Y=amt_col, values=risk rate for each metric flag column.
    Risk is calculated as events/base where:
      base = count of non-null values in metric column
      events = sum of metric values with NaN treated as 0
    """
    if metrics is None:
        metrics = ["D4P6_FLG", "D4P9_FLG", "D4P13_FLG"]

    # basic checks
    need_cols = ["TEST_NAME", "TEST_GROUP_NAME", term_col, amt_col]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # filter one test + one group
    d = df.loc[
        (df["TEST_NAME"].eq(test_name)) &
        (df["TEST_GROUP_NAME"].eq(test_group_name))
    ].copy()

    if d.empty:
        raise ValueError("No rows after filtering by TEST_NAME and TEST_GROUP_NAME.")

    # metric name sanity check (common case: D4P12 instead of D4P13)
    for m in metrics:
        if m not in d.columns:
            hint = ""
            if m.upper() == "D4P13_FLG" and "D4P12_FLG" in d.columns:
                hint = " (Hint: you likely want 'D4P12_FLG' instead of 'D4P13_FLG')"
            raise ValueError(f"Metric column '{m}' not found in df.{hint}")

    # ensure term is numeric-ish if possible
    # (we don't force convert, just sorting)
    term_order = _sort_terms(d[term_col])
    amt_order = _sort_amt_groups(d[amt_col])

    pivots = {}

    for metric in metrics:
        tmp = d[[term_col, amt_col, metric]].copy()
        tmp["base"] = tmp[metric].notna().astype(int)
        tmp["events"] = tmp[metric].fillna(0).astype(float)

        agg = (
            tmp.groupby([amt_col, term_col], dropna=False)
               .agg(base=("base", "sum"), events=("events", "sum"))
               .reset_index()
        )
        agg["rate"] = np.where(agg["base"] > 0, agg["events"] / agg["base"], np.nan)

        pivot = agg.pivot(index=amt_col, columns=term_col, values="rate")
        # reindex to stable order
        pivot = pivot.reindex(index=amt_order)
        pivot = pivot.reindex(columns=term_order)

        pivots[metric] = pivot

        # ---- plot heatmap ----
        fig, ax = plt.subplots(figsize=(1.2 * max(6, len(pivot.columns)), 0.5 * max(6, len(pivot.index))))
        data = pivot.values

        im = ax.imshow(data, aspect="auto")

        ax.set_title(f"{test_name} / {test_group_name} — {metric} risk\nY={amt_col}, X={term_col}")
        ax.set_xlabel(term_col)
        ax.set_ylabel(amt_col)

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        # annotate cells with %
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v*100:.1f}%", ha="center", va="center", fontsize=8)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Risk rate")

        plt.tight_layout()
        plt.show()

    return pivots

# ---- Example usage ----
# NOTE: if you don't have D4P13_FLG, replace it with D4P12_FLG
pivots = risk_heatmap_by_term_and_amt(
    df,
    test_name="RBP_AVG",
    test_group_name="avg_good",
    term_col="REQUESTED_TERM",
    amt_col="MAXLOANAMTGROUP",
    metrics=["D4P6_FLG", "D4P9_FLG", "D4P13_FLG"],  # or ["D4P6_FLG","D4P9_FLG","D4P12_FLG"]
)


import pandas as pd
import matplotlib.pyplot as plt

def plot_risk_heatmap(
    df: pd.DataFrame,
    *,
    test_name: str,
    test_group_name: str,
    metric_col: str,
    term_col: str = "REQUESTED_TERM",
    amt_col: str = "MAX_LOAN_AMT_GR",
    term_order=(6, 9, 12, 15, 18),
    amt_order=("<=10K", "<=25K", "<=35K", "<=45K", "<=55K"),
    extra_filters: dict | None = None,
    agg="mean",   # or "mean" if metric already 0/1, or custom
    title_prefix: str | None = None,
):
    d = df.copy()

    # --- filters ---
    d = d[(d["TEST_NAME"] == test_name) & (d["TEST_GROUP_NAME"] == test_group_name)]
    if extra_filters:
        for col, val in extra_filters.items():
            if isinstance(val, (list, tuple, set)):
                d = d[d[col].isin(list(val))]
            else:
                d = d[d[col].eq(val)]

    # --- enforce category order for axes ---
    d[term_col] = pd.Categorical(d[term_col], categories=list(term_order), ordered=True)
    d[amt_col]  = pd.Categorical(d[amt_col],  categories=list(amt_order),  ordered=True)

    # --- risk aggregation (assumes metric is 0/1 flag; mean = rate) ---
    if agg == "mean":
        risk = d.groupby([amt_col, term_col], observed=True)[metric_col].mean()
    else:
        risk = d.groupby([amt_col, term_col], observed=True)[metric_col].agg(agg)

    pivot = risk.reset_index().pivot(index=amt_col, columns=term_col, values=metric_col)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # green(low) -> red(high): use "RdYlGn_r" (reversed)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns])

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(y) for y in pivot.index])

    ax.set_xlabel(term_col)
    ax.set_ylabel(amt_col)

    # annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v*100:.1f}%", ha="center", va="center", fontsize=9)

    title = f"{test_name} / {test_group_name} — {metric_col} risk\nY={amt_col}, X={term_col}"
    if title_prefix:
        title = f"{title_prefix}\n{title}"
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Risk rate")

    plt.tight_layout()
    return pivot, fig, ax


# --- Example ---
# pivot, fig, ax = plot_risk_heatmap(
#     df,
#     test_name="RBP_GOOD",
#     test_group_name="good_supgood",
#     metric_col="D4P12_FLG",
#     term_col="REQUESTED_TERM",
#     amt_col="MAX_LOAN_AMT_GR",
#     amt_order=("<=10K","<=25K","<=35K","<=45K","<=55K"),
#     term_order=(6,9,12,15,18),
# )
# plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

def plot_risk_heatmap_with_n(
    df: pd.DataFrame,
    *,
    test_name: str,
    test_group_name: str,
    metric_col: str,
    term_col: str = "REQUESTED_TERM",
    amt_col: str = "MAX_LOAN_AMT_GR",
    term_order=(6, 9, 12, 15, 18),
    amt_order=("<=10K", "<=25K", "<=35K", "<=45K", "<=55K"),
    extra_filters: dict | None = None,
    # --- color scale control ---
    scale_mode: str = "clip_quantile",   # "clip_quantile" | "fixed" | "log" | "power"
    clip_q: float = 0.95,                # used for clip_quantile
    vmin: float | None = None,           # used for fixed / optional for others
    vmax: float | None = None,           # used for fixed / optional for others
    power_gamma: float = 0.6,            # used for power
    show_n: bool = True,
    min_n_to_annotate: int = 1,           # show risk only if n >= this
):
    d = df.copy()
    d = d[(d["TEST_NAME"] == test_name) & (d["TEST_GROUP_NAME"] == test_group_name)]

    if extra_filters:
        for col, val in extra_filters.items():
            if isinstance(val, (list, tuple, set)):
                d = d[d[col].isin(list(val))]
            else:
                d = d[d[col].eq(val)]

    # enforce axis ordering
    d[term_col] = pd.Categorical(d[term_col], categories=list(term_order), ordered=True)
    d[amt_col]  = pd.Categorical(d[amt_col],  categories=list(amt_order),  ordered=True)

    # risk (mean of 0/1 flag) and n
    g = d.groupby([amt_col, term_col], observed=True)
    out = g[metric_col].agg(risk="mean", n="size").reset_index()

    risk_pivot = out.pivot(index=amt_col, columns=term_col, values="risk")
    n_pivot    = out.pivot(index=amt_col, columns=term_col, values="n")

    Z = risk_pivot.values.astype(float)

    # --- normalization to avoid extreme values dominating palette ---
    finite = np.isfinite(Z)
    Z_f = Z[finite]

    if scale_mode == "clip_quantile":
        _vmin = np.nanmin(Z_f) if vmin is None else vmin
        _vmax = np.nanquantile(Z_f, clip_q) if vmax is None else vmax
        norm = colors.Normalize(vmin=_vmin, vmax=_vmax, clip=True)

    elif scale_mode == "fixed":
        if vmin is None or vmax is None:
            raise ValueError("For scale_mode='fixed' please set both vmin and vmax (e.g. 0 and 0.35).")
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    elif scale_mode == "log":
        # log needs positive vmin; add small epsilon if zeros exist
        eps = 1e-6
        _vmin = max(eps, (np.nanmin(Z_f) if vmin is None else vmin))
        _vmax = np.nanmax(Z_f) if vmax is None else vmax
        norm = colors.LogNorm(vmin=_vmin, vmax=_vmax, clip=True)

    elif scale_mode == "power":
        _vmin = np.nanmin(Z_f) if vmin is None else vmin
        _vmax = np.nanmax(Z_f) if vmax is None else vmax
        norm = colors.PowerNorm(gamma=power_gamma, vmin=_vmin, vmax=_vmax, clip=True)

    else:
        raise ValueError("Unknown scale_mode. Use: clip_quantile | fixed | log | power")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    im = ax.imshow(Z, aspect="auto", cmap="RdYlGn_r", norm=norm)

    ax.set_xticks(range(len(risk_pivot.columns)))
    ax.set_xticklabels([str(x) for x in risk_pivot.columns])
    ax.set_yticks(range(len(risk_pivot.index)))
    ax.set_yticklabels([str(y) for y in risk_pivot.index])
    ax.set_xlabel(term_col)
    ax.set_ylabel(amt_col)

    ax.set_title(f"{test_name} / {test_group_name} — {metric_col} risk\nY={amt_col}, X={term_col}")

    # annotate: risk + n
    for i in range(risk_pivot.shape[0]):
        for j in range(risk_pivot.shape[1]):
            v = Z[i, j]
            n = n_pivot.values[i, j] if n_pivot is not None else np.nan
            if np.isfinite(v) and (pd.isna(n) or n >= min_n_to_annotate):
                txt = f"{v*100:.1f}%"
                if show_n and pd.notna(n):
                    txt += f"\n(n={int(n)})"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Risk rate")

    plt.tight_layout()
    return risk_pivot, n_pivot, fig, ax
    
    
risk_p, n_p, fig, ax = plot_risk_heatmap_with_n(
    df,
    test_name="RBP_GOOD",
    test_group_name="good_good",
    metric_col="D4P12_FLG",
    amt_col="MAX_LOAN_AMT_GR",
    term_col="REQUESTED_TERM",
    amt_order=("<=10K","<=25K","<=35K","<=45K","<=55K"),
    term_order=(6,9,12,15,18),
    scale_mode="clip_quantile",
    clip_q=0.95,
)
plt.show()



import numpy as np
import pandas as pd

# ============================================================
# CONFIG: rename these column names to match your dataframe
# ============================================================
COL = {
    # experiment
    "test_name": "TEST_NAME",
    "test_group": "TEST_GROUP_NAME",

    # unique offer id
    "offer_id": "OFFER_RK",

    # dates
    "offer_create_dt": "offer_creation_dttm",
    "app_create_dt": "app_creation_dttm",
    "app_form_filled_dt": "app_form_filled_dttm",
    "util_dt": "utilization_dttm",

    # finance
    "npv": "NPV",                     # NPV value (may appear multiple rows per offer)
    "disb_amt": "DISBURSEMENT_AMT",   # financial amount / issuance amount (0 if not issued)
}

# ============================================================
# Helpers
# ============================================================
def _to_dt(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    return pd.to_datetime(s, errors="coerce")

def _days_diff(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later - earlier).dt.days

def _conv_from_start_to_event(start_dt, event_dt, now, window_days):
    age = _days_diff(now, start_dt)
    diff = _days_diff(event_dt, start_dt)
    return np.where(
        age <= window_days,
        np.nan,
        np.where(event_dt.notna() & (diff <= window_days), 1.0, 0.0),
    )

def _conv_from_app_to_event(app_dt, event_dt, now, window_days):
    age = _days_diff(now, app_dt)
    diff = _days_diff(event_dt, app_dt)
    return np.where(
        app_dt.isna(),
        np.nan,
        np.where(
            age <= window_days,
            np.nan,
            np.where(event_dt.notna() & (diff <= window_days), 1.0, 0.0),
        ),
    )

# ============================================================
# Main: KPIs for ONE test and TWO groups only
# ============================================================
def compute_kpis_for_test_two_groups(
    df: pd.DataFrame,
    test_name: str,
    group_a: str,
    group_b: str,
    col=COL,
    now=None,
) -> pd.DataFrame:
    d = df.copy()

    # filter to one test + two groups
    mask = (
        d[col["test_name"]].astype(str).eq(str(test_name)) &
        d[col["test_group"]].astype(str).isin([str(group_a), str(group_b)])
    )
    d = d.loc[mask].copy()
    if d.empty:
        raise ValueError("No rows after filtering by test_name and two groups. Check names/columns.")

    # datetimes
    d[col["offer_create_dt"]] = _to_dt(d[col["offer_create_dt"]])
    d[col["app_create_dt"]] = _to_dt(d[col["app_create_dt"]])
    d[col["app_form_filled_dt"]] = _to_dt(d[col["app_form_filled_dt"]])
    d[col["util_dt"]] = _to_dt(d[col["util_dt"]])

    now = pd.Timestamp.now().normalize() if now is None else pd.Timestamp(now).normalize()

    # conversions (row-level)
    d["conv_offer_to_new_app_1m"] = _conv_from_start_to_event(
        d[col["offer_create_dt"]], d[col["app_create_dt"]], now=now, window_days=30
    )
    d["conv_offer_to_full_app_1m"] = _conv_from_start_to_event(
        d[col["offer_create_dt"]], d[col["app_form_filled_dt"]], now=now, window_days=30
    )
    d["conv_offer_to_utilization_1m"] = _conv_from_start_to_event(
        d[col["offer_create_dt"]], d[col["util_dt"]], now=now, window_days=30
    )
    d["conv_offer_to_utilization_2m"] = _conv_from_start_to_event(
        d[col["offer_create_dt"]], d[col["util_dt"]], now=now, window_days=60
    )
    d["conv_new_app_to_utilization_5d"] = _conv_from_app_to_event(
        d[col["app_create_dt"]], d[col["util_dt"]], now=now, window_days=5
    )
    d["conv_new_app_to_full_app_5d"] = _conv_from_app_to_event(
        d[col["app_create_dt"]], d[col["app_form_filled_dt"]], now=now, window_days=5
    )

    conv_cols = [
        "conv_offer_to_new_app_1m",
        "conv_offer_to_full_app_1m",
        "conv_offer_to_utilization_1m",
        "conv_offer_to_utilization_2m",
        "conv_new_app_to_utilization_5d",
        "conv_new_app_to_full_app_5d",
    ]

    # offer-level aggregation (handles multiple rows per offer)
    agg_dict = {
        col["npv"]: "sum",
        col["disb_amt"]: "sum",
        col["util_dt"]: lambda x: int(pd.Series(x).notna().any()),
    }
    for c in conv_cols:
        agg_dict[c] = lambda x: pd.Series(x).max(skipna=True)

    offer_df = (
        d[[col["offer_id"], col["test_name"], col["test_group"], col["npv"], col["disb_amt"], col["util_dt"]] + conv_cols]
        .groupby([col["offer_id"], col["test_name"], col["test_group"]], dropna=False, as_index=False)
        .agg(agg_dict)
        .rename(columns={col["util_dt"]: "UTIL_FLG"})
    )

    def _rate(s: pd.Series) -> float:
        s2 = s.dropna()
        return float(s2.mean()) if len(s2) else np.nan

    # group-level KPIs
    out = []
    for grp_name, gdf in offer_df.groupby(col["test_group"], dropna=False):
        npv_total = float(gdf[col["npv"]].sum(skipna=True))
        n_offers = int(gdf.shape[0])
        n_util = int(gdf["UTIL_FLG"].sum(skipna=True))
        fin_total = float(gdf[col["disb_amt"]].sum(skipna=True))

        row = {
            "TEST_NAME": test_name,
            "TEST_GROUP_NAME": grp_name,
            "NPV_TOTAL": npv_total,
            "NPV_PER_OFFER": (npv_total / n_offers) if n_offers else np.nan,
            "NPV_PER_UTIL": (npv_total / n_util) if n_util else np.nan,
            "NPV_PER_FIN_AMOUNT": (npv_total / fin_total) if fin_total else np.nan,
            "N_OFFERS": n_offers,
            "N_UTIL_OFFERS": n_util,
            "FIN_AMOUNT_TOTAL": fin_total,
        }
        for c in conv_cols:
            row[c] = _rate(gdf[c])

        out.append(row)

    res = pd.DataFrame(out)

    # Ensure output order is exactly group_a then group_b (if present)
    order = [group_a, group_b]
    res["__ord"] = res["TEST_GROUP_NAME"].apply(lambda x: order.index(x) if x in order else 999)
    res = res.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)

    return res


# ============================================================
# USAGE
# ============================================================
# kpis = compute_kpis_for_test_two_groups(
#     df,
#     test_name="RBP_GOOD",
#     group_a="good_basic",
#     group_b="good_good",
# )
# display(kpis)



import pandas as pd
import numpy as np

def make_hashable_id(x):
    # bytes/bytearray -> hex string
    if isinstance(x, (bytes, bytearray)):
        return x.hex()
    # numpy bytes -> decode
    if isinstance(x, np.bytes_):
        return bytes(x).hex()
    # list/array -> stringify (на всякий случай)
    if isinstance(x, (list, tuple, np.ndarray)):
        return str(x)
    return x

def fix_groupby_keys(df: pd.DataFrame, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(make_hashable_id)
    return df

# Пример: прогоняем ключи, которые используешь в groupby
key_cols = ["offer_id", "test_name", "test_group"]   # добавь сюда person_rk/agreement_rk если тоже группируешь
df2_fixed = fix_groupby_keys(df2, key_cols)

# дальше твой groupby уже не упадет
# offer_df = df2_fixed.groupby(key_cols, dropna=False, as_index=False).agg(...)

import numpy as np
import pandas as pd
from scipy import stats

# ---------- helpers ----------
def _welch_ttest(a: pd.Series, b: pd.Series):
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue

def _two_prop_ztest(success_a, n_a, success_b, n_b):
    # returns p-value for H0: p_a == p_b
    if n_a == 0 or n_b == 0:
        return np.nan
    p_pool = (success_a + success_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    if se == 0:
        return np.nan
    z = (success_a/n_a - success_b/n_b) / se
    return 2 * (1 - stats.norm.cdf(abs(z)))

def add_significance_for_two_groups(
    df: pd.DataFrame,
    kpis: pd.DataFrame,
    test_name: str,
    group_a: str,
    group_b: str,
    cols: dict,
    conv_cols: list,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    df: raw offer-level dataframe
    kpis: output of compute_kpis_for_test_two_groups (2 rows: group_a & group_b)
    cols: mapping of column names in df
      required keys:
        'offer_id', 'test_name', 'test_group', 'npv', 'fin_amount', 'util_dttm'
    conv_cols: list of conversion flag columns already computed in df (0/1 or bool or NULL)
    """

    # filter to one test and two groups
    d = df.loc[
        df[cols["test_name"]].eq(test_name) &
        df[cols["test_group"]].isin([group_a, group_b])
    ].copy()

    # one row per offer_id (важно чтобы не дублировались офферы)
    # npv / fin_amount — берём первое/сумму? обычно 1 запись на offer_id; если дубль — берём max/first
    agg = {
        cols["npv"]: "sum",              # если NPV по офферу может быть по строкам — суммируем
        cols["fin_amount"]: "sum",       # аналогично
        cols["util_dttm"]: lambda x: x.notna().any(),  # флаг утилизации
    }
    for c in conv_cols:
        agg[c] = lambda x: pd.Series(x).dropna().astype(int).max() if pd.Series(x).dropna().shape[0] else 0

    offer = (
        d.groupby([cols["offer_id"], cols["test_name"], cols["test_group"]], as_index=False)
         .agg(agg)
         .rename(columns={cols["util_dttm"]: "UTIL_FLG"})
    )

    # derived per-offer metrics for tests
    offer["NPV_PER_OFFER_ROW"] = offer[cols["npv"]]
    offer["NPV_PER_FIN_AMOUNT_ROW"] = offer[cols["npv"]] / offer[cols["fin_amount"]].replace({0: np.nan})
    offer["NPV_PER_UTIL_ROW"] = np.where(offer["UTIL_FLG"], offer[cols["npv"]], np.nan)

    # split groups
    A = offer.loc[offer[cols["test_group"]].eq(group_a)]
    B = offer.loc[offer[cols["test_group"]].eq(group_b)]

    # p-values
    p_npv_per_offer = _welch_ttest(A["NPV_PER_OFFER_ROW"], B["NPV_PER_OFFER_ROW"])
    p_npv_per_util  = _welch_ttest(A["NPV_PER_UTIL_ROW"],  B["NPV_PER_UTIL_ROW"])
    p_npv_per_fin   = _welch_ttest(A["NPV_PER_FIN_AMOUNT_ROW"], B["NPV_PER_FIN_AMOUNT_ROW"])

    # conversion p-values
    conv_pvals = {}
    for c in conv_cols:
        # конверсия по офферам: success = sum(flag), n = #offers
        succ_a, n_a = int(A[c].sum()), int(len(A))
        succ_b, n_b = int(B[c].sum()), int(len(B))
        conv_pvals[c] = _two_prop_ztest(succ_a, n_a, succ_b, n_b)

    # attach to KPI table (в виде отдельных колонок)
    out = kpis.copy()

    # метрики как в твоей таблице
    # (NPV_TOTAL — без pvalue)
    out["PVAL_NPV_PER_OFFER"] = p_npv_per_offer
    out["PVAL_NPV_PER_UTIL"]  = p_npv_per_util
    out["PVAL_NPV_PER_FIN_AMOUNT"] = p_npv_per_fin

    # add conversions p-values
    for c, pv in conv_pvals.items():
        out[f"PVAL_{c}"] = pv

    # significance flags
    out["SIG_NPV_PER_OFFER"] = out["PVAL_NPV_PER_OFFER"].lt(alpha)
    out["SIG_NPV_PER_UTIL"]  = out["PVAL_NPV_PER_UTIL"].lt(alpha)
    out["SIG_NPV_PER_FIN_AMOUNT"] = out["PVAL_NPV_PER_FIN_AMOUNT"].lt(alpha)
    for c in conv_cols:
        out[f"SIG_{c}"] = out[f"PVAL_{c}"].lt(alpha)

    return out


# ---------- Example usage ----------
# cols = {
#   "offer_id": "OFFER_RK",
#   "test_name": "TEST_NAME",
#   "test_group": "TEST_GROUP_NAME",
#   "npv": "NPV",
#   "fin_amount": "FIN_AMOUNT",      # сумма выдачи
#   "util_dttm": "UTILIZATION_DTTM", # дата утилизации
# }
# conv_cols = ["CONV_OFFER_TO_UTILIZATION_2M"]  # и любые другие конверсии, которые ты считаешь
#
# kpis = compute_kpis_for_test_two_groups(df, test_name="RBP_GOOD", group_a="good_basic", group_b="good_good")
# kpis_sig = add_significance_for_two_groups(df, kpis, "RBP_GOOD", "good_basic", "good_good", cols, conv_cols, alpha=0.05)
# display(kpis_sig)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# -----------------------------
# Helper: order for MAX_LOAN_AMT_GR like "<=10K", "<=25K", ... (and keeps unknowns at the end)
# -----------------------------
def _parse_amt_bucket(x: str):
    if pd.isna(x):
        return np.inf
    s = str(x).strip().upper().replace(" ", "")
    # examples: "<=10K", "<=35K", "<=55K", "10-25K"
    # take the biggest number in the string as bucket "upper bound" proxy
    import re
    nums = re.findall(r"(\d+)", s)
    if not nums:
        return np.inf
    return float(nums[-1])

def _ensure_order(values, explicit_order=None):
    vals = [v for v in pd.unique(values) if pd.notna(v)]
    if explicit_order is not None:
        # keep only those present, keep original explicit order
        return [v for v in explicit_order if v in set(vals)]
    # otherwise sort by numeric parsing
    return sorted(vals, key=_parse_amt_bucket)

# -----------------------------
# Main function
# -----------------------------
def plot_risk_by_limit_multi(
    df: pd.DataFrame,
    test_col: str = "TEST_NAME",
    group_col: str = "TEST_GROUP_NAME",
    amt_col: str = "MAX_LOAN_AMT_GR",
    risk_col: str = "D4P6_FLG",     # <- choose any: "D4P6_FLG", "D4P9_FLG", "D4P12_FLG", etc.
    tests=None,                     # list[str] or None => all
    groups=None,                    # list[str] or None => all
    amt_order=None,                 # explicit order list, optional
    min_n_per_point: int = 30,      # hide points with too small n
    ci: bool = True,                # draw 95% CI (Wilson) for binomial risk
    alpha: float = 0.05,
    normalize_y_as_percent: bool = True,
    title: str | None = None,
    figsize=(12, 6),
):
    """
    Plots risk vs limit bucket with multiple lines for (test, group).
    Risk is treated as binomial mean of risk_col (0/1 or boolean).
    """

    d = df.copy()

    # filter tests/groups
    if tests is not None:
        d = d[d[test_col].isin(tests)]
    if groups is not None:
        d = d[d[group_col].isin(groups)]

    # clean risk to numeric 0/1
    d = d[pd.notna(d[risk_col]) & pd.notna(d[amt_col]) & pd.notna(d[test_col]) & pd.notna(d[group_col])]
    d[risk_col] = d[risk_col].astype(float)

    if d.empty:
        raise ValueError("No data after filters. Check tests/groups/risk_col/amt_col.")

    # category order for amount
    order = _ensure_order(d[amt_col], explicit_order=amt_order)
    d[amt_col] = pd.Categorical(d[amt_col], categories=order, ordered=True)

    # aggregate
    agg = (
        d.groupby([test_col, group_col, amt_col], dropna=False)
        .agg(n=(risk_col, "size"), risk=(risk_col, "mean"))
        .reset_index()
    )

    # Wilson CI for proportions (more stable than normal approx)
    def wilson_ci(p, n, z=1.96):
        if n <= 0 or pd.isna(p):
            return (np.nan, np.nan)
        denom = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denom
        half = (z * np.sqrt((p*(1-p) + z**2/(4*n)) / n)) / denom
        return center - half, center + half

    if ci:
        z = 1.96  # ~95%
        lows, highs = [], []
        for p, n in zip(agg["risk"].values, agg["n"].values):
            lo, hi = wilson_ci(p, int(n), z=z)
            lows.append(lo); highs.append(hi)
        agg["ci_low"] = lows
        agg["ci_high"] = highs

    # mask small n
    agg.loc[agg["n"] < min_n_per_point, ["risk", "ci_low", "ci_high"]] = np.nan

    # plotting
    fig, ax = plt.subplots(figsize=figsize)

    # x positions (categorical)
    x_labels = order
    x_pos = np.arange(len(x_labels))

    # build lines for each (test, group)
    line_keys = agg[[test_col, group_col]].drop_duplicates().sort_values([test_col, group_col]).values.tolist()

    for t, g in line_keys:
        sub = agg[(agg[test_col] == t) & (agg[group_col] == g)].sort_values(amt_col)
        # align to full x axis
        sub = sub.set_index(amt_col).reindex(x_labels)
        y = sub["risk"].values.astype(float)
        label = f"{t} / {g}"

        ax.plot(x_pos, y, marker="o", linewidth=2, label=label)

        if ci:
            lo = sub["ci_low"].values.astype(float)
            hi = sub["ci_high"].values.astype(float)
            yerr_low = y - lo
            yerr_high = hi - y
            # avoid negative yerr when NaN
            yerr = np.vstack([np.where(np.isfinite(yerr_low), yerr_low, np.nan),
                              np.where(np.isfinite(yerr_high), yerr_high, np.nan)])
            ax.errorbar(x_pos, y, yerr=yerr, fmt="none", capsize=3)

        # annotate n under points
        for i, nval in enumerate(sub["n"].values):
            if pd.isna(y[i]) or pd.isna(nval):
                continue
            ax.annotate(f"n={int(nval)}", (x_pos[i], y[i]),
                        textcoords="offset points", xytext=(0, -18),
                        ha="center", fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(x) for x in x_labels], rotation=0)
    ax.set_xlabel(amt_col)

    ax.set_ylabel(f"Risk ({risk_col})")
    if normalize_y_as_percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.grid(True, axis="y", alpha=0.3)

    if title is None:
        title = f"Risk vs {amt_col} (metric={risk_col})"
    ax.set_title(title)

    ax.legend(loc="best")
    plt.tight_layout()

    return agg, fig, ax


# -----------------------------
# Example usage
# -----------------------------
# agg, fig, ax = plot_risk_by_limit_multi(
#     df,
#     test_col="TEST_NAME",
#     group_col="TEST_GROUP_NAME",
#     amt_col="MAX_LOAN_AMT_GR",
#     risk_col="D4P12_FLG",
#     tests=["RBP_GOOD", "RBP_AVG"],
#     groups=["good_basic", "good_good"],
#     amt_order=["<=10K","<=25K","<=35K","<=45K","<=55K"],   # optional
#     min_n_per_point=30,
#     ci=True,
#     title="RBP: D4P12 risk vs limit",
# )
# plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

def plot_risk_by_limit_two_metrics_one_group(
    df: pd.DataFrame,
    test_name: str,
    group_name: str,
    metric_cols: list,                 # например ["D4P6_FLG", "D4P12_FLG"]
    test_col: str = "TEST_NAME",
    group_col: str = "TEST_GROUP_NAME",
    limit_col: str = "MAX_LOAN_AMT_GR",
    limit_order: list | None = None,   # например ["<=10K","<=25K","<=35K","<=45K","<=55K"]
    alpha: float = 0.05,
    min_n_per_point: int = 30,
    ci: bool = True,
    title: str | None = None,
):
    # 1) filter
    d = df.loc[(df[test_col] == test_name) & (df[group_col] == group_name)].copy()
    if d.empty:
        raise ValueError("No rows after filtering by test/group.")

    # 2) order for x
    if limit_order is None:
        limit_order = sorted(d[limit_col].dropna().unique().tolist())

    d[limit_col] = pd.Categorical(d[limit_col], categories=limit_order, ordered=True)

    # 3) aggregate for each metric separately
    out = []
    for m in metric_cols:
        tmp = (
            d.groupby(limit_col, dropna=False)[m]
            .agg(n="count", k="sum")  # k=число плохих (если флаг 0/1)
            .reset_index()
        )
        tmp["metric"] = m
        out.append(tmp)

    agg = pd.concat(out, ignore_index=True)

    # 4) CI (Wilson)
    if ci:
        lo, hi = proportion_confint(
            count=agg["k"].astype(int).values,
            nobs=agg["n"].astype(int).values,
            alpha=alpha,
            method="wilson",
        )
        agg["ci_low"] = lo
        agg["ci_high"] = hi

    # 5) drop small n points
    agg = agg[agg["n"] >= min_n_per_point].copy()

    # 6) plot
    x_labels = limit_order
    x_pos_map = {lab: i for i, lab in enumerate(x_labels)}

    fig, ax = plt.subplots(figsize=(10, 5))

    for m in metric_cols:
        sub = agg[agg["metric"] == m].copy()
        if sub.empty:
            continue

        sub["x"] = sub[limit_col].astype(str).map(x_pos_map)
        sub = sub.sort_values("x")

        y = (sub["k"] / sub["n"]).astype(float).values
        x = sub["x"].values

        line, = ax.plot(x, y, marker="o", linewidth=2, label=m)
        c = line.get_color()

        # CI
        if ci and ("ci_low" in sub.columns):
            lo = sub["ci_low"].astype(float).values
            hi = sub["ci_high"].astype(float).values
            yerr = np.vstack([y - lo, hi - y])
            ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=3, ecolor=c, elinewidth=1.5)

        # n labels
        for xi, yi, ni in zip(x, y, sub["n"].astype(int).values):
            ax.annotate(f"n={ni}", (xi, yi), textcoords="offset points", xytext=(0, -12),
                        ha="center", fontsize=9, color=c)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Risk rate")
    ax.set_xlabel(limit_col)

    if title is None:
        title = f"{test_name} / {group_name}: risk vs limit ({', '.join(metric_cols)})"
    ax.set_title(title)

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return agg, fig, ax
    
    
    
agg, fig, ax = plot_risk_by_limit_two_metrics_one_group(
    df,
    test_name="RBP_GOOD",
    group_name="good_good",
    metric_cols=["D4P6_FLG", "D4P12_FLG"],
    limit_order=["<=10K","<=25K","<=35K","<=45K","<=55K"],
    min_n_per_point=30,
    ci=True,
    alpha=0.05
)