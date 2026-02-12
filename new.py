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