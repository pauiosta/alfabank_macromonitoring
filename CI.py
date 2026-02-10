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