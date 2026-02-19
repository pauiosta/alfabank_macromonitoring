with base as (
  select
    -- ключевые метрики
    ag.fpd_7plus_flg,
    ag.d2p3_flg,
    ag.d4p6_flg,
    ag.d4p9_flg,
    ag.d4p12_flg,
    pd.pd_pm_01_xsell_cl_cl_v2 as pd_score,
    n.npv,

    -- finstar flags
    finstar_hit_app_found_flg,
    finstar_hit_app_not_found_flg,
    finstar_hit_app_found_flg_dpd10,
    finstar_hit_app_found_flg_last_reject
  from <YOUR_BASE_TABLE_OR_CTE>
)

select
  bucket,
  count(*) as n,
  avg(fpd_7plus_flg::int) as fpd7_rate,
  avg(d2p3_flg::int)      as d2p3_rate,
  avg(d4p6_flg::int)      as d4p6_rate,
  avg(d4p9_flg::int)      as d4p9_rate,
  avg(d4p12_flg::int)     as d4p12_rate,
  avg(pd_score)           as avg_pd,
  avg(npv)                as avg_npv,
  sum(npv)                as total_npv
from (
  select 'hit_app_found' as bucket, * from base where finstar_hit_app_found_flg = 1
  union all
  select 'hit_app_not_found' as bucket, * from base where finstar_hit_app_not_found_flg = 1
  union all
  select 'hit_dpd10_last3m' as bucket, * from base where finstar_hit_app_found_flg_dpd10 = 1
  union all
  select 'hit_last_reject_1y' as bucket, * from base where finstar_hit_app_found_flg_last_reject = 1
) t
group by 1
order by n desc;





with base as (
  select
    bucket,
    type,  -- YES / NO
    fpd_7plus_flg,
    d2p3_flg,
    d4p6_flg,
    d4p9_flg,
    d4p12_flg,
    pd_pm_01_xsell_cl_cl_v2 as pd_score,
    npv
  from <YOUR_BUCKETED_DATASET>
),
agg as (
  select
    bucket,
    type,
    count(*) as num_obs,

    -- maturity (non-null) + events
    count(fpd_7plus_flg) as fpd7_matured_n,
    sum(fpd_7plus_flg)   as fpd7_events_n,

    count(d2p3_flg) as d2p3_matured_n,
    sum(d2p3_flg)   as d2p3_events_n,

    count(d4p6_flg) as d4p6_matured_n,
    sum(d4p6_flg)   as d4p6_events_n,

    count(d4p9_flg) as d4p9_matured_n,
    sum(d4p9_flg)   as d4p9_events_n,

    count(d4p12_flg) as d4p12_matured_n,
    sum(d4p12_flg)   as d4p12_events_n,

    avg(pd_score) as avg_pd,
    avg(npv)      as avg_npv,
    sum(npv)      as total_npv

  from base
  group by 1,2
),
final as (
  select
    *,
    1.959963984540054::float as z
  from agg
)

select
  bucket,
  type,
  num_obs,

  /* ===================== FPD7+ ===================== */
  fpd7_matured_n,
  fpd7_events_n,
  (fpd7_events_n / nullif(fpd7_matured_n,0)) * 100 as fpd7_rate_pct,

  (
    (
      (fpd7_events_n / nullif(fpd7_matured_n,0))
      + (z*z) / (2*fpd7_matured_n)
    ) / (1 + (z*z)/fpd7_matured_n)
    -
    z * sqrt(
      ((fpd7_events_n / nullif(fpd7_matured_n,0)) * (1 - (fpd7_events_n / nullif(fpd7_matured_n,0))) / fpd7_matured_n)
      + (z*z)/(4*fpd7_matured_n*fpd7_matured_n)
    ) / (1 + (z*z)/fpd7_matured_n)
  ) * 100 as fpd7_ci_lo_pct,

  (
    (
      (fpd7_events_n / nullif(fpd7_matured_n,0))
      + (z*z) / (2*fpd7_matured_n)
    ) / (1 + (z*z)/fpd7_matured_n)
    +
    z * sqrt(
      ((fpd7_events_n / nullif(fpd7_matured_n,0)) * (1 - (fpd7_events_n / nullif(fpd7_matured_n,0))) / fpd7_matured_n)
      + (z*z)/(4*fpd7_matured_n*fpd7_matured_n)
    ) / (1 + (z*z)/fpd7_matured_n)
  ) * 100 as fpd7_ci_hi_pct,

  /* ===================== D2P3 ===================== */
  d2p3_matured_n,
  d2p3_events_n,
  (d2p3_events_n / nullif(d2p3_matured_n,0)) * 100 as d2p3_rate_pct,

  (
    (
      (d2p3_events_n / nullif(d2p3_matured_n,0))
      + (z*z) / (2*d2p3_matured_n)
    ) / (1 + (z*z)/d2p3_matured_n)
    -
    z * sqrt(
      ((d2p3_events_n / nullif(d2p3_matured_n,0)) * (1 - (d2p3_events_n / nullif(d2p3_matured_n,0))) / d2p3_matured_n)
      + (z*z)/(4*d2p3_matured_n*d2p3_matured_n)
    ) / (1 + (z*z)/d2p3_matured_n)
  ) * 100 as d2p3_ci_lo_pct,

  (
    (
      (d2p3_events_n / nullif(d2p3_matured_n,0))
      + (z*z) / (2*d2p3_matured_n)
    ) / (1 + (z*z)/d2p3_matured_n)
    +
    z * sqrt(
      ((d2p3_events_n / nullif(d2p3_matured_n,0)) * (1 - (d2p3_events_n / nullif(d2p3_matured_n,0))) / d2p3_matured_n)
      + (z*z)/(4*d2p3_matured_n*d2p3_matured_n)
    ) / (1 + (z*z)/d2p3_matured_n)
  ) * 100 as d2p3_ci_hi_pct,

  /* ===================== D4P6 ===================== */
  d4p6_matured_n,
  d4p6_events_n,
  (d4p6_events_n / nullif(d4p6_matured_n,0)) * 100 as d4p6_rate_pct,

  (
    (
      (d4p6_events_n / nullif(d4p6_matured_n,0))
      + (z*z) / (2*d4p6_matured_n)
    ) / (1 + (z*z)/d4p6_matured_n)
    -
    z * sqrt(
      ((d4p6_events_n / nullif(d4p6_matured_n,0)) * (1 - (d4p6_events_n / nullif(d4p6_matured_n,0))) / d4p6_matured_n)
      + (z*z)/(4*d4p6_matured_n*d4p6_matured_n)
    ) / (1 + (z*z)/d4p6_matured_n)
  ) * 100 as d4p6_ci_lo_pct,

  (
    (
      (d4p6_events_n / nullif(d4p6_matured_n,0))
      + (z*z) / (2*d4p6_matured_n)
    ) / (1 + (z*z)/d4p6_matured_n)
    +
    z * sqrt(
      ((d4p6_events_n / nullif(d4p6_matured_n,0)) * (1 - (d4p6_events_n / nullif(d4p6_matured_n,0))) / d4p6_matured_n)
      + (z*z)/(4*d4p6_matured_n*d4p6_matured_n)
    ) / (1 + (z*z)/d4p6_matured_n)
  ) * 100 as d4p6_ci_hi_pct,

  /* ===================== D4P9 ===================== */
  d4p9_matured_n,
  d4p9_events_n,
  (d4p9_events_n / nullif(d4p9_matured_n,0)) * 100 as d4p9_rate_pct,

  (
    (
      (d4p9_events_n / nullif(d4p9_matured_n,0))
      + (z*z) / (2*d4p9_matured_n)
    ) / (1 + (z*z)/d4p9_matured_n)
    -
    z * sqrt(
      ((d4p9_events_n / nullif(d4p9_matured_n,0)) * (1 - (d4p9_events_n / nullif(d4p9_matured_n,0))) / d4p9_matured_n)
      + (z*z)/(4*d4p9_matured_n*d4p9_matured_n)
    ) / (1 + (z*z)/d4p9_matured_n)
  ) * 100 as d4p9_ci_lo_pct,

  (
    (
      (d4p9_events_n / nullif(d4p9_matured_n,0))
      + (z*z) / (2*d4p9_matured_n)
    ) / (1 + (z*z)/d4p9_matured_n)
    +
    z * sqrt(
      ((d4p9_events_n / nullif(d4p9_matured_n,0)) * (1 - (d4p9_events_n / nullif(d4p9_matured_n,0))) / d4p9_matured_n)
      + (z*z)/(4*d4p9_matured_n*d4p9_matured_n)
    ) / (1 + (z*z)/d4p9_matured_n)
  ) * 100 as d4p9_ci_hi_pct,

  /* ===================== D4P12 ===================== */
  d4p12_matured_n,
  d4p12_events_n,
  (d4p12_events_n / nullif(d4p12_matured_n,0)) * 100 as d4p12_rate_pct,

  (
    (
      (d4p12_events_n / nullif(d4p12_matured_n,0))
      + (z*z) / (2*d4p12_matured_n)
    ) / (1 + (z*z)/d4p12_matured_n)
    -
    z * sqrt(
      ((d4p12_events_n / nullif(d4p12_matured_n,0)) * (1 - (d4p12_events_n / nullif(d4p12_matured_n,0))) / d4p12_matured_n)
      + (z*z)/(4*d4p12_matured_n*d4p12_matured_n)
    ) / (1 + (z*z)/d4p12_matured_n)
  ) * 100 as d4p12_ci_lo_pct,

  (
    (
      (d4p12_events_n / nullif(d4p12_matured_n,0))
      + (z*z) / (2*d4p12_matured_n)
    ) / (1 + (z*z)/d4p12_matured_n)
    +
    z * sqrt(
      ((d4p12_events_n / nullif(d4p12_matured_n,0)) * (1 - (d4p12_events_n / nullif(d4p12_matured_n,0))) / d4p12_matured_n)
      + (z*z)/(4*d4p12_matured_n*d4p12_matured_n)
    ) / (1 + (z*z)/d4p12_matured_n)
  ) * 100 as d4p12_ci_hi_pct,

  avg_pd,
  avg_npv,
  total_npv
from final
order by bucket, type;