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