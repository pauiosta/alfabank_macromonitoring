import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats.proportion import proportion_confint, proportions_ztest


def _term_to_metric(term: int) -> str:
    # можно переопределить под свои названия
    return f"D4P{int(term)}_FLG"


def plot_term_aligned_risk(
    df: pd.DataFrame,
    test_name: str,
    groups: list[str],                      # например ["good_basic", "good_good"]
    term_col: str = "REQUESTED_TERM",
    test_col: str = "TEST_NAME",
    group_col: str = "TEST_GROUP_NAME",
    term_order: list[int] | None = None,    # например [6,9,12,15,18]
    alpha: float = 0.05,
    min_n_per_point: int = 30,
    ci: bool = True,
    show_n: bool = True,
    title: str | None = None,
    pvalue_between_two_groups: bool = True, # если groups ровно 2 — добавит табличку p-values
):
    d = df.loc[df[test_col].eq(test_name)].copy()
    if d.empty:
        raise ValueError(f"No rows for test={test_name}")

    # term_order
    if term_order is None:
        term_order = sorted(pd.Series(d[term_col].dropna().unique()).astype(int).tolist())
    term_order = [int(t) for t in term_order]

    # соберём "aligned" датасет: для каждой строки возьмём нужную метрику по term
    needed_metrics = {_term_to_metric(t) for t in term_order}
    existing_metrics = [m for m in needed_metrics if m in d.columns]
    if not existing_metrics:
        raise ValueError(
            f"No metric columns found. Expected something like: {sorted(list(needed_metrics))[:6]} ..."
        )

    # берем значения метрики по строке (по term)
    # аккуратно: если метрики нет в df — будет NaN
    d["_ALIGNED_METRIC"] = d[term_col].astype(int).map(_term_to_metric)
    d["_ALIGNED_VALUE"] = np.nan
    for m in existing_metrics:
        mask = d["_ALIGNED_METRIC"].eq(m)
        d.loc[mask, "_ALIGNED_VALUE"] = d.loc[mask, m].astype(float)

    # фильтр по группам
    d = d[d[group_col].isin(groups)].copy()
    if d.empty:
        raise ValueError("No rows after filtering by groups.")

    # агрегация: k/n по group x term
    agg = (
        d.groupby([group_col, term_col], dropna=False)["_ALIGNED_VALUE"]
        .agg(n="count", k="sum")
        .reset_index()
    )
    agg[term_col] = agg[term_col].astype(int)

    # риск
    agg["risk"] = agg["k"] / agg["n"]

    # CI (Wilson)
    if ci:
        lo, hi = proportion_confint(
            count=agg["k"].astype(int).values,
            nobs=agg["n"].astype(int).values,
            alpha=alpha,
            method="wilson",
        )
        agg["ci_low"] = lo
        agg["ci_high"] = hi

    # drop small n
    agg = agg[agg["n"] >= min_n_per_point].copy()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    x_map = {t: i for i, t in enumerate(term_order)}

    for g in groups:
        sub = agg[agg[group_col].eq(g)].copy()
        if sub.empty:
            continue
        sub["x"] = sub[term_col].map(x_map)
        sub = sub.sort_values("x")

        x = sub["x"].values
        y = sub["risk"].values

        line, = ax.plot(x, y, marker="o", linewidth=2, label=g)
        c = line.get_color()

        if ci and "ci_low" in sub.columns:
            lo = sub["ci_low"].values
            hi = sub["ci_high"].values
            yerr = np.vstack([y - lo, hi - y])
            ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=3, ecolor=c, elinewidth=1.5)

        if show_n:
            for xi, yi, ni in zip(x, y, sub["n"].astype(int).values):
                ax.annotate(f"n={ni}", (xi, yi), textcoords="offset points",
                            xytext=(0, -12), ha="center", fontsize=9, color=c)

    ax.set_xticks(range(len(term_order)))
    ax.set_xticklabels(term_order)
    ax.set_xlabel(term_col)
    ax.set_ylabel("Term-aligned risk: D4P(term)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    if title is None:
        title = f"{test_name}: term-aligned risk (D4P(term))"
    ax.set_title(title)

    plt.tight_layout()
    plt.show()

    # --- Optional: p-values table for 2 groups ---
    ptab = None
    if pvalue_between_two_groups and len(groups) == 2:
        gA, gB = groups
        rows = []
        for t in term_order:
            a = agg[(agg[group_col] == gA) & (agg[term_col] == t)]
            b = agg[(agg[group_col] == gB) & (agg[term_col] == t)]
            if a.empty or b.empty:
                continue

            k = np.array([int(a["k"].iloc[0]), int(b["k"].iloc[0])])
            n = np.array([int(a["n"].iloc[0]), int(b["n"].iloc[0])])

            # two-sided z-test for proportions
            stat, pval = proportions_ztest(count=k, nobs=n, alternative="two-sided")
            ra = a["risk"].iloc[0]
            rb = b["risk"].iloc[0]
            uplift = (rb - ra) / ra if ra > 0 else np.nan

            rows.append({
                term_col: t,
                f"risk_{gA}": ra,
                f"risk_{gB}": rb,
                "uplift_rel": uplift,
                "p_value": pval,
                f"n_{gA}": n[0],
                f"n_{gB}": n[1],
            })
        ptab = pd.DataFrame(rows)

    return agg, ptab, fig, ax
    
    
    
agg, ptab, fig, ax = plot_term_aligned_risk(
    df,
    test_name="RBP_GOOD",
    groups=["good_basic", "good_good"],
    term_order=[6, 9, 12, 15, 18],
    min_n_per_point=30,
    ci=True,
    alpha=0.05,
)
display(ptab)  # если хочешь таблицу с p-values по каждому term



import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def _term_to_metric(term: int) -> str:
    return f"D4P{int(term)}_FLG"

def corr_term_vs_risk_table(
    df: pd.DataFrame,
    test_col: str = "TEST_NAME",
    group_col: str = "TEST_GROUP_NAME",
    term_col: str = "REQUESTED_TERM",
    mode: str = "term_aligned",     # "term_aligned" | "fixed_metric"
    risk_col: str | None = None,    # нужно только для mode="fixed_metric"
    min_n_per_term: int = 30,       # минимальный размер на точку term
    min_terms: int = 3,             # минимум разных term чтобы считать корреляцию
) -> pd.DataFrame:
    d = df.copy()
    d[term_col] = pd.to_numeric(d[term_col], errors="coerce").astype("Int64")

    if mode == "fixed_metric":
        if not risk_col:
            raise ValueError("For mode='fixed_metric' you must provide risk_col, e.g. 'D4P12_FLG'")
        if risk_col not in d.columns:
            raise ValueError(f"Column '{risk_col}' not found in df")
        d["_RISK"] = pd.to_numeric(d[risk_col], errors="coerce")

    elif mode == "term_aligned":
        # выбираем метрику по term и складываем в одну колонку _RISK
        d["_METRIC"] = d[term_col].astype("Int64").map(lambda x: _term_to_metric(x) if pd.notna(x) else np.nan)
        d["_RISK"] = np.nan
        # заполняем _RISK из соответствующих колонок, которые реально есть в df
        metrics_needed = d["_METRIC"].dropna().unique().tolist()
        for m in metrics_needed:
            if m in d.columns:
                mask = d["_METRIC"].eq(m)
                d.loc[mask, "_RISK"] = pd.to_numeric(d.loc[mask, m], errors="coerce")
    else:
        raise ValueError("mode must be 'term_aligned' or 'fixed_metric'")

    # агрегируем: для каждого test/group/term считаем risk_rate и n
    agg = (
        d.dropna(subset=[test_col, group_col, term_col, "_RISK"])
         .groupby([test_col, group_col, term_col], dropna=False)["_RISK"]
         .agg(n="count", risk_rate="mean")
         .reset_index()
    )

    # фильтр по min_n_per_term
    agg = agg[agg["n"] >= min_n_per_term].copy()

    # теперь корреляция по точкам (term -> risk_rate) внутри test/group
    rows = []
    for (test, grp), sub in agg.groupby([test_col, group_col], dropna=False):
        sub = sub.sort_values(term_col)
        if sub[term_col].nunique() < min_terms:
            continue

        x = sub[term_col].astype(float).to_numpy()
        y = sub["risk_rate"].astype(float).to_numpy()

        # Pearson
        pr, pp = pearsonr(x, y)
        # Spearman
        sr, sp = spearmanr(x, y)

        rows.append({
            test_col: test,
            group_col: grp,
            "n_terms": int(sub[term_col].nunique()),
            "sum_n": int(sub["n"].sum()),
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_rho": sr,
            "spearman_p": sp,
        })

    out = pd.DataFrame(rows).sort_values([test_col, group_col]).reset_index(drop=True)
    return out, agg


# ===== Example usage =====
# 1) Term-aligned (рекомендую)
corr_tbl, term_points = corr_term_vs_risk_table(
    df,
    mode="term_aligned",
    min_n_per_term=30,
    min_terms=3,
)

display(corr_tbl)

# 2) Если хочешь корреляцию именно D4P12 vs term
# corr_tbl_12, term_points_12 = corr_term_vs_risk_table(
#     df,
#     mode="fixed_metric",
#     risk_col="D4P12_FLG",
#     min_n_per_term=30,
#     min_terms=3,
# )
# display(corr_tbl_12)


import numpy as np
import pandas as pd

# -----------------------------
# 1) Utility: make nice PD buckets
# -----------------------------
def add_pd_buckets(
    df: pd.DataFrame,
    pd_col: str,
    method: str = "quantile",   # "quantile" | "fixed"
    n_bins: int = 10,           # for quantile
    fixed_edges=None,           # for fixed, e.g. [0,0.02,0.04,0.06,0.08,0.10,1.0]
    labels=None,
    out_col: str = "PD_BUCKET",
):
    d = df.copy()
    d[pd_col] = pd.to_numeric(d[pd_col], errors="coerce")

    if method == "quantile":
        # qcut may fail if many equal values -> duplicates="drop"
        d[out_col] = pd.qcut(d[pd_col], q=n_bins, duplicates="drop")
    elif method == "fixed":
        if fixed_edges is None:
            raise ValueError("For method='fixed' you must provide fixed_edges")
        if labels is None:
            # auto labels like "0-2%", "2-4%", ...
            labels = []
            for a, b in zip(fixed_edges[:-1], fixed_edges[1:]):
                labels.append(f"{a:.2%}–{b:.2%}")
        d[out_col] = pd.cut(d[pd_col], bins=fixed_edges, labels=labels, include_lowest=True, right=True)
    else:
        raise ValueError("method must be 'quantile' or 'fixed'")

    return d


# -----------------------------
# 2) Crosstab PD x Amount with:
#    - n
#    - mean risk (optional)
#    - mean PD (optional)
# -----------------------------
def crosstab_pd_x_amount(
    df: pd.DataFrame,
    pd_col: str,
    amount_col: str,
    risk_col: str | None = None,           # e.g. "D4P12_FLG"
    pd_bucket_col: str = "PD_BUCKET",
    amount_order: list | None = None,      # optional explicit order for amount buckets
    min_n: int = 0,                        # optionally blank out low-N cells
):
    d = df.copy()

    # keep needed cols
    keep = [pd_col, amount_col, pd_bucket_col]
    if risk_col:
        keep.append(risk_col)
    d = d[keep].copy()

    # clean
    d[pd_col] = pd.to_numeric(d[pd_col], errors="coerce")
    if risk_col:
        d[risk_col] = pd.to_numeric(d[risk_col], errors="coerce")
    d = d.dropna(subset=[pd_bucket_col, amount_col])

    # amount order if provided
    if amount_order is not None:
        d[amount_col] = pd.Categorical(d[amount_col], categories=amount_order, ordered=True)

    # N table
    n_tbl = pd.crosstab(d[pd_bucket_col], d[amount_col], dropna=False)

    # mean PD per cell
    mean_pd = (
        d.pivot_table(index=pd_bucket_col, columns=amount_col, values=pd_col, aggfunc="mean", dropna=False)
    )

    # risk per cell (if provided)
    if risk_col:
        risk_tbl = (
            d.pivot_table(index=pd_bucket_col, columns=amount_col, values=risk_col, aggfunc="mean", dropna=False)
        )
    else:
        risk_tbl = None

    # blank out low-N cells if requested
    if min_n and min_n > 0:
        mask_low = n_tbl < min_n
        mean_pd = mean_pd.mask(mask_low)
        if risk_tbl is not None:
            risk_tbl = risk_tbl.mask(mask_low)

    return n_tbl, mean_pd, risk_tbl


# -----------------------------
# 3) Example usage
# -----------------------------
# df = ... your dataframe
#
# 3.1) Add PD buckets (quantiles / deciles)
# df_b = add_pd_buckets(df, pd_col="PD_SCORE", method="quantile", n_bins=10, out_col="PD_BUCKET")
#
# 3.2) Crosstab PD bucket x amount bucket
# n_tbl, mean_pd_tbl, risk_tbl = crosstab_pd_x_amount(
#     df_b,
#     pd_col="PD_SCORE",
#     amount_col="MAX_LOAN_AMT_GR",
#     risk_col="D4P12_FLG",                 # or None if you only want counts/PD
#     pd_bucket_col="PD_BUCKET",
#     amount_order=["<=10K","<=25K","<=35K","<=45K","<=55K"],  # optional
#     min_n=30
# )
#
# display(n_tbl)
# display(mean_pd_tbl)
# display(risk_tbl)   # this is your key: actual risk by PD x amount


import numpy as np
import pandas as pd


def pd_bucket_profile(
    df: pd.DataFrame,
    *,
    test_name: str,
    group_name: str,
    test_col: str = "TEST_NAME",
    group_col: str = "TEST_GROUP_NAME",
    pd_col: str = "PD",                      # PD score/probability column
    numeric_cols: list[str] | None = None,   # which numeric metrics to average (term, amount, payment, etc.)
    risk_cols: list[str] | None = None,      # risk indicators (e.g. ["D4P6_FLG","D4P9_FLG","D4P12_FLG"])
    n_buckets: int = 10,
    bucket_edges: list[float] | None = None, # optional explicit edges instead of qcut
    bucket_labels: list[str] | None = None,  # optional labels
    dropna_pd: bool = True,
) -> pd.DataFrame:
    """
    Splits PD into buckets for a выбранного test/group and computes mean of given numeric columns,
    plus means of risk columns (risk rate), and counts.

    - If bucket_edges is provided -> uses pd.cut with these edges.
    - Else -> uses pd.qcut into n_buckets quantiles.
    """

    d = df.loc[df[test_col].eq(test_name) & df[group_col].eq(group_name)].copy()

    if dropna_pd:
        d = d[d[pd_col].notna()].copy()

    # Validate / auto-pick numeric columns
    if numeric_cols is None:
        # take all numeric cols except obvious IDs and risk cols
        risk_cols_set = set(risk_cols or [])
        exclude = {test_col, group_col}
        cand = [
            c for c in d.columns
            if c not in exclude
            and c != pd_col
            and c not in risk_cols_set
            and pd.api.types.is_numeric_dtype(d[c])
        ]
        numeric_cols = cand

    if risk_cols is None:
        risk_cols = []

    # Build PD buckets
    if bucket_edges is not None:
        bins = np.array(bucket_edges, dtype=float)
        if bucket_labels is None:
            bucket_labels = [f"[{bins[i]:.4g}, {bins[i+1]:.4g})" for i in range(len(bins) - 1)]
        d["PD_BUCKET"] = pd.cut(d[pd_col], bins=bins, labels=bucket_labels, include_lowest=True, right=False)
    else:
        # qcut can fail if too many duplicates; handle with rank
        x = d[pd_col]
        try:
            d["PD_BUCKET"] = pd.qcut(x, q=n_buckets, duplicates="drop")
        except ValueError:
            # fallback: rank then qcut
            r = x.rank(method="average")
            d["PD_BUCKET"] = pd.qcut(r, q=n_buckets, duplicates="drop")

    # Aggregations: mean numeric metrics + mean risk flags (risk rate) + counts
    agg_dict = {c: "mean" for c in numeric_cols}
    for rc in risk_cols:
        agg_dict[rc] = "mean"  # assuming flags 0/1 -> mean = risk rate

    out = (
        d.groupby("PD_BUCKET", dropna=False)
         .agg(
            N=("PD_BUCKET", "size"),
            PD_MEAN=(pd_col, "mean"),
            PD_MIN=(pd_col, "min"),
            PD_MAX=(pd_col, "max"),
            **agg_dict
         )
         .reset_index()
    )

    # Add metadata columns
    out.insert(0, "TEST_NAME", test_name)
    out.insert(1, "TEST_GROUP_NAME", group_name)

    # Nice formatting helpers (optional)
    return out


# ---------------- Example usage ----------------
# You choose what to average:
# term/amount/payment + any other numeric columns you want
metrics_to_avg = ["REQUESTED_TERM", "DISB_AMT", "MONTHLY_PAYMENT"]  # <-- rename to your dataset columns

# Risk flags you want to compare (means = rates)
risk_flags = ["D4P6_FLG", "D4P9_FLG", "D4P12_FLG"]  # <-- rename to your dataset columns

profile = pd_bucket_profile(
    df,
    test_name="RBP_GOOD",
    group_name="good_good",
    test_col="TEST_NAME",
    group_col="TEST_GROUP_NAME",
    pd_col="PD_SCORE",              # <-- your PD column
    numeric_cols=metrics_to_avg,
    risk_cols=risk_flags,
    n_buckets=10
)

display(profile)

import matplotlib.pyplot as plt

col = "regular_payment_amt"

# 1) Базовая статистика
display(df[col].describe(percentiles=[.01,.05,.1,.25,.5,.75,.9,.95,.99]))
print("NaN:", df[col].isna().sum(), " / total:", len(df))
print("<=0:", (df[col] <= 0).sum())

# 2) Гистограмма (сырое распределение)
x = df[col].dropna()
plt.figure()
plt.hist(x, bins=50)
plt.title(f"Distribution: {col}")
plt.xlabel(col)
plt.ylabel("count")
plt.show()

# 3) Если очень перекошено/длинный хвост — гистограмма в лог-шкале по X
x_pos = x[x > 0]
plt.figure()
plt.hist(x_pos, bins=50)
plt.xscale("log")
plt.title(f"Distribution (log-x): {col} (x>0)")
plt.xlabel(col)
plt.ylabel("count")
plt.show()





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_npv_heatmap_with_n(
    df: pd.DataFrame,
    *,
    test_name: str,
    test_group_name: str,
    x_col: str,                 # например "REQUESTED_TERM"
    y_col: str,                 # например "REGULAR_PAYMENT_AMT_GR"
    npv_col: str = "NPV",       # твой столбец NPV
    mode: str = "per_offer",    # "total" | "per_offer" | "per_util" | "per_fin_amount"
    offer_id_col: str = "OFFER_RK",
    util_dt_col: str = "UTILIZATION_DTTM",
    fin_amt_col: str = "DISB_AMT",
    test_col: str = "TEST_NAME",
    group_col: str = "TEST_GROUP_NAME",
    x_order=None,
    y_order=None,
    min_n_per_cell: int = 1,
    scale_mode: str = "clip_quantile",   # "none" | "clip_quantile"
    clip_q: float = 0.95,
    cmap: str = "RdYlGn",        # NPV: больше = лучше -> зелёный
    title: str | None = None,
    fmt_value: str = "{:.0f}",   # формат NPV в ячейке
):
    d = df.loc[df[test_col].eq(test_name) & df[group_col].eq(test_group_name)].copy()
    d = d[d[x_col].notna() & d[y_col].notna()].copy()

    # n в ячейке: сколько офферов (если есть offer_id_col), иначе строк
    if offer_id_col in d.columns:
        n_series = d.groupby([y_col, x_col])[offer_id_col].nunique()
    else:
        n_series = d.groupby([y_col, x_col]).size()

    # числитель: сумма NPV
    num = d.groupby([y_col, x_col])[npv_col].sum()

    # знаменатель
    if mode == "total":
        val = num
        value_label = "NPV total"
    elif mode == "per_offer":
        denom = d.groupby([y_col, x_col])[offer_id_col].nunique()
        val = num / denom.replace(0, np.nan)
        value_label = "NPV per offer"
    elif mode == "per_util":
        util_flag = d[util_dt_col].notna().astype(int)
        denom = util_flag.groupby([d[y_col], d[x_col]]).sum()
        val = num / denom.replace(0, np.nan)
        value_label = "NPV per util"
    elif mode == "per_fin_amount":
        denom = d.groupby([y_col, x_col])[fin_amt_col].sum()
        val = num / denom.replace(0, np.nan)
        value_label = "NPV / financial amount"
    else:
        raise ValueError("mode must be one of: total, per_offer, per_util, per_fin_amount")

    pivot_v = val.unstack(x_col)
    pivot_n = n_series.unstack(x_col)

    # порядок осей
    if x_order is not None:
        pivot_v = pivot_v.reindex(columns=list(x_order))
        pivot_n = pivot_n.reindex(columns=list(x_order))
    if y_order is not None:
        pivot_v = pivot_v.reindex(index=list(y_order))
        pivot_n = pivot_n.reindex(index=list(y_order))

    # фильтр по min_n_per_cell
    pivot_v = pivot_v.where(pivot_n >= min_n_per_cell)

    # клип шкалы, чтобы выбросы не "ломали" палитру
    v_for_scale = pivot_v.to_numpy().astype(float)
    v_for_scale = v_for_scale[~np.isnan(v_for_scale)]
    vmin, vmax = (None, None)
    if scale_mode == "clip_quantile" and v_for_scale.size > 0:
        vmax = np.quantile(v_for_scale, clip_q)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot_v.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(pivot_v.shape[1]))
    ax.set_yticks(np.arange(pivot_v.shape[0]))
    ax.set_xticklabels(pivot_v.columns)
    ax.set_yticklabels(pivot_v.index)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    if title is None:
        title = f"{test_name} / {test_group_name} — {value_label}\nY={y_col}, X={x_col}"
    ax.set_title(title)

    # подписи: value + (n=)
    for i in range(pivot_v.shape[0]):
        for j in range(pivot_v.shape[1]):
            v = pivot_v.iat[i, j]
            n = pivot_n.iat[i, j]
            if pd.notna(v) and pd.notna(n):
                ax.text(j, i, f"{fmt_value.format(v)}\n(n={int(n)})",
                        ha="center", va="center", fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(value_label)

    plt.tight_layout()
    return pivot_v, pivot_n, fig, ax
    
    
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_npv_heatmap_with_n(
    df: pd.DataFrame,
    *,
    test_name: str,
    test_group_name: str,
    x_col: str,                 # например "REQUESTED_TERM"
    y_col: str,                 # например "REGULAR_PAYMENT_AMT_GR"
    npv_col: str = "NPV",       # твой столбец NPV
    mode: str = "per_offer",    # "total" | "per_offer" | "per_util" | "per_fin_amount"
    offer_id_col: str = "OFFER_RK",
    util_dt_col: str = "UTILIZATION_DTTM",
    fin_amt_col: str = "DISB_AMT",
    test_col: str = "TEST_NAME",
    group_col: str = "TEST_GROUP_NAME",
    x_order=None,
    y_order=None,
    min_n_per_cell: int = 1,
    scale_mode: str = "clip_quantile",   # "none" | "clip_quantile"
    clip_q: float = 0.95,
    cmap: str = "RdYlGn",        # NPV: больше = лучше -> зелёный
    title: str | None = None,
    fmt_value: str = "{:.0f}",   # формат NPV в ячейке
):
    d = df.loc[df[test_col].eq(test_name) & df[group_col].eq(test_group_name)].copy()
    d = d[d[x_col].notna() & d[y_col].notna()].copy()

    # n в ячейке: сколько офферов (если есть offer_id_col), иначе строк
    if offer_id_col in d.columns:
        n_series = d.groupby([y_col, x_col])[offer_id_col].nunique()
    else:
        n_series = d.groupby([y_col, x_col]).size()

    # числитель: сумма NPV
    num = d.groupby([y_col, x_col])[npv_col].sum()

    # знаменатель
    if mode == "total":
        val = num
        value_label = "NPV total"
    elif mode == "per_offer":
        denom = d.groupby([y_col, x_col])[offer_id_col].nunique()
        val = num / denom.replace(0, np.nan)
        value_label = "NPV per offer"
    elif mode == "per_util":
        util_flag = d[util_dt_col].notna().astype(int)
        denom = util_flag.groupby([d[y_col], d[x_col]]).sum()
        val = num / denom.replace(0, np.nan)
        value_label = "NPV per util"
    elif mode == "per_fin_amount":
        denom = d.groupby([y_col, x_col])[fin_amt_col].sum()
        val = num / denom.replace(0, np.nan)
        value_label = "NPV / financial amount"
    else:
        raise ValueError("mode must be one of: total, per_offer, per_util, per_fin_amount")

    pivot_v = val.unstack(x_col)
    pivot_n = n_series.unstack(x_col)

    # порядок осей
    if x_order is not None:
        pivot_v = pivot_v.reindex(columns=list(x_order))
        pivot_n = pivot_n.reindex(columns=list(x_order))
    if y_order is not None:
        pivot_v = pivot_v.reindex(index=list(y_order))
        pivot_n = pivot_n.reindex(index=list(y_order))

    # фильтр по min_n_per_cell
    pivot_v = pivot_v.where(pivot_n >= min_n_per_cell)

    # клип шкалы, чтобы выбросы не "ломали" палитру
    v_for_scale = pivot_v.to_numpy().astype(float)
    v_for_scale = v_for_scale[~np.isnan(v_for_scale)]
    vmin, vmax = (None, None)
    if scale_mode == "clip_quantile" and v_for_scale.size > 0:
        vmax = np.quantile(v_for_scale, clip_q)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot_v.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(pivot_v.shape[1]))
    ax.set_yticks(np.arange(pivot_v.shape[0]))
    ax.set_xticklabels(pivot_v.columns)
    ax.set_yticklabels(pivot_v.index)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    if title is None:
        title = f"{test_name} / {test_group_name} — {value_label}\nY={y_col}, X={x_col}"
    ax.set_title(title)

    # подписи: value + (n=)
    for i in range(pivot_v.shape[0]):
        for j in range(pivot_v.shape[1]):
            v = pivot_v.iat[i, j]
            n = pivot_n.iat[i, j]
            if pd.notna(v) and pd.notna(n):
                ax.text(j, i, f"{fmt_value.format(v)}\n(n={int(n)})",
                        ha="center", va="center", fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(value_label)

    plt.tight_layout()
    return pivot_v, pivot_n, fig, ax


pivot_v, pivot_n, fig, ax = plot_npv_heatmap_with_n(
    df,
    test_name="RBP_GOOD",
    test_group_name="good_good",
    x_col="REQUESTED_TERM",
    y_col="REGULAR_PAYMENT_AMT_GR",
    npv_col="NPV",
    mode="per_offer",  # попробуй также "total" / "per_util" / "per_fin_amount"
    x_order=[6, 9, 12, 15, 18],
    y_order=[">4000", "<=4000", "<=3000", "<=2000", "<=1000"],
    min_n_per_cell=20,
    scale_mode="clip_quantile",
    clip_q=0.95,
    cmap="RdYlGn",  # зелёный = высокий NPV
)
plt.show()


import numpy as np
import pandas as pd


def pd_bucket_profile(
    df: pd.DataFrame,
    *,
    test_name: str,
    group_name: str,
    test_col: str = "TEST_NAME",
    group_col: str = "TEST_GROUP_NAME",
    pd_col: str = "PD",
    numeric_cols: list[str] | None = None,   # числовые метрики для среднего (term, amount, payment, etc.)
    risk_cols: list[str] | None = None,      # флаги риска 0/1, mean = risk rate
    n_buckets: int = 10,
    bucket_edges: list[float] | None = None, # если хочешь фиксированные границы
    dropna_pd: bool = True,
) -> pd.DataFrame:

    d = df.loc[df[test_col].eq(test_name) & df[group_col].eq(group_name)].copy()
    if dropna_pd:
        d = d[d[pd_col].notna()].copy()

    if risk_cols is None:
        risk_cols = []

    # если numeric_cols не задан — возьмём все числовые (кроме pd/test/group/risk)
    if numeric_cols is None:
        exclude = {test_col, group_col, pd_col, *risk_cols}
        numeric_cols = [
            c for c in d.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(d[c])
        ]

    # ---------- buckets ----------
    if bucket_edges is not None:
        bins = np.array(bucket_edges, dtype=float)
        d["PD_BUCKET"] = pd.cut(d[pd_col], bins=bins, include_lowest=True, right=False)
    else:
        try:
            d["PD_BUCKET"] = pd.qcut(d[pd_col], q=n_buckets, duplicates="drop")
        except ValueError:
            # fallback если много одинаковых значений
            r = d[pd_col].rank(method="average")
            d["PD_BUCKET"] = pd.qcut(r, q=n_buckets, duplicates="drop")

    # ---------- aggregations (ВАЖНО: named aggregation) ----------
    agg_kwargs = {
        "N": ("PD_BUCKET", "size"),
        "PD_MEAN": (pd_col, "mean"),
        "PD_MIN": (pd_col, "min"),
        "PD_MAX": (pd_col, "max"),
    }

    # средние по выбранным метрикам
    for c in numeric_cols:
        agg_kwargs[f"AVG_{c}"] = (c, "mean")

    # risk flags: mean = rate
    for rc in risk_cols:
        agg_kwargs[f"RISK_{rc}"] = (rc, "mean")

    out = (
        d.groupby("PD_BUCKET", dropna=False)
         .agg(**agg_kwargs)
         .reset_index()
    )

    out.insert(0, "TEST_NAME", test_name)
    out.insert(1, "TEST_GROUP_NAME", group_name)
    return out


# ----------- Example usage -----------
metrics_to_avg = ["REQUESTED_TERM", "DISB_AMT", "MONTHLY_PAYMENT"]  # переименуешь под себя
risk_flags = ["D4P6_FLG", "D4P9_FLG", "D4P12_FLG"]                 # переименуешь под себя

profile = pd_bucket_profile(
    df,
    test_name="RBP_GOOD",
    group_name="good_good",
    test_col="TEST_NAME",
    group_col="TEST_GROUP_NAME",
    pd_col="PD_PM_01_XSELL_CL_CL_V2",
    numeric_cols=metrics_to_avg,
    risk_cols=risk_flags,
    n_buckets=10
)

display(profile)


import numpy as np
import pandas as pd

def compare_d4p6_segments(
    df: pd.DataFrame,
    *,
    d4p6_col: str = "D4P6_FLG",
    metrics_to_avg = ("REQUESTED_TERM", "DISB_AMT", "MONTHLY_PAYMENT"),
    test_col: str | None = None,
    group_col: str | None = None,
    test_name: str | None = None,
    group_name: str | None = None,
):
    d = df.copy()

    # optional filter by test/group
    if test_col and test_name is not None:
        d = d[d[test_col].eq(test_name)]
    if group_col and group_name is not None:
        d = d[d[group_col].eq(group_name)]

    # keep only 0/1 rows
    d = d[d[d4p6_col].isin([0, 1])].copy()

    # build summary
    out = []
    for flag, g in d.groupby(d4p6_col, dropna=False):
        row = {
            d4p6_col: int(flag),
            "N_ROWS": int(len(g)),
        }
        for m in metrics_to_avg:
            s = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_MEAN"] = float(s.mean(skipna=True))
            row[f"{m}_MEDIAN"] = float(s.median(skipna=True))
            row[f"{m}_STD"] = float(s.std(skipna=True))
            row[f"{m}_N"] = int(s.notna().sum())
        out.append(row)

    res = pd.DataFrame(out).sort_values(d4p6_col).reset_index(drop=True)

    # add "diff" row: (1 - 0) for MEANs/ MEDIANs
    if set(res[d4p6_col]) == {0, 1}:
        r0 = res[res[d4p6_col] == 0].iloc[0]
        r1 = res[res[d4p6_col] == 1].iloc[0]
        diff = {d4p6_col: "DIFF (1-0)", "N_ROWS": int(r1["N_ROWS"] - r0["N_ROWS"])}
        for m in metrics_to_avg:
            diff[f"{m}_MEAN"]   = float(r1[f"{m}_MEAN"] - r0[f"{m}_MEAN"])
            diff[f"{m}_MEDIAN"] = float(r1[f"{m}_MEDIAN"] - r0[f"{m}_MEDIAN"])
            diff[f"{m}_STD"]    = np.nan
            diff[f"{m}_N"]      = int(r1[f"{m}_N"] - r0[f"{m}_N"])
        res = pd.concat([res, pd.DataFrame([diff])], ignore_index=True)

    return res


# ===== Example usage =====
metrics_to_avg = ["REQUESTED_TERM", "DISB_AMT", "MONTHLY_PAYMENT"]

tbl = compare_d4p6_segments(
    df,
    d4p6_col="D4P6_FLG",
    metrics_to_avg=metrics_to_avg,
    # если нужно ограничить одним тестом/группой — раскомментируй:
    # test_col="TEST_NAME", test_name="RBP_GOOD",
    # group_col="TEST_GROUP_NAME", group_name="good_good",
)

display(tbl)