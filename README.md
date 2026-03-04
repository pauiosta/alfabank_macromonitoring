Business Value

We analyzed Finstar application-hit rules and found clear value asymmetry across subsegments: two buckets concentrate unpriced risk + negative NPV, while one bucket looks risk-neutral with positive NPV and can be used to grow volumes (especially in MRB=1).

Key findings (by segment)

CLN → CLN
	•	Baseline (no hit): FPD7+ 5.2%, D2P3 7.7%, D4P6 11.1%, Avg NPV 1,966 (Total NPV 16,978,260)
	•	finstar_hit_app_not_found: FPD7+ 4.1%, D2P3 7.0%, D4P6 11.5%, Avg NPV 1,934 (Total NPV 1,224,265) → risk comparable / NPV positive
	•	finstar_hit_app_found_last_reject: FPD7+ 10.7%, D2P3 18.2%, D4P6 22.4%, Avg NPV -106 (Total NPV -51,698) → high risk + negative NPV
	•	finstar_hit_app_found_dpd10: FPD7+ 17.8%, D2P3 25.4%, D4P6 34.3%, Avg NPV -1,350 (Total NPV -282,088) → very high risk + strongly negative NPV
	•	Estimated portfolio impact if we limit disbursements in the two worst buckets: risk 12.2% → 10.5% and NPV 1,024 → 1,625 (CLN→CLN).

POS → CLN
	•	Baseline: D4P6 9.2%, Avg NPV 1,344 (Total NPV 147,644,596)
	•	finstar_hit_app_not_found: D4P6 11.5%, Avg NPV 1,021 (Total NPV 5,705,293) → still positive NPV
	•	finstar_hit_app_found_last_reject: D4P6 21.1%, Avg NPV -144 (Total NPV -462,235) → negative NPV
	•	finstar_hit_app_found_dpd10: D4P6 23.8%, Avg NPV -811 (Total NPV -364,250) → negative NPV
	•	Estimated impact if we limit disbursements in the two worst buckets: risk 11.4% → 10.7% and NPV 810 → 999 (POS→CLN).
	•	For MRB=1 (projection): finstar_hit_app_not_found D4P6 ≈ 16.15%, NPV ≈ 334, while finstar_hit_app_found D4P6 ≈ 21.32%, NPV ≈ -882 → supports opening app_not_found in MRB=1 and keeping other buckets restricted.

Xsell → CLN
	•	Baseline: D4P6 7.3%, Avg NPV 1,818 (Total NPV 76,346,909)
	•	finstar_hit_app_not_found: D4P6 8.8%, Avg NPV 1,335 (Total NPV 4,588,321) → risk close to baseline, NPV positive
	•	finstar_hit_app_found_last_reject: D4P6 16.0%, Avg NPV -1,091 (Total NPV -3,261,075) → negative NPV
	•	finstar_hit_app_found_dpd10: D4P6 17.7%, Avg NPV -2,384 (Total NPV -1,234,981) → negative NPV
	•	Estimated impact if we limit disbursements in the two worst buckets: risk 8.9% → 8.3% and NPV 1,865 → 2,214 (Xsell→CLN).

Conclusion:
	•	Close finstar_hit_app_found_last_reject + finstar_hit_app_found_dpd10 for MRB=0 because they drive high realized delinquency and negative NPV (risk not fully captured by PD).
	•	Open finstar_hit_app_not_found for MRB=1 to gain additional offers/disbursements with acceptable risk and positive NPV.

⸻

Current Behavior

We currently have 4 Finstar application-hit rules (names used in scoring / rules engine):
	1.	Finstar. Hit, application was found → finstar_hit_app_found
	2.	Finstar. Hit, application was not found → finstar_hit_app_not_found
	3.	Finstar. Hit, application found, no agreement, last reject <= 1 year ago → finstar_hit_app_found_last_reject
	4.	Finstar. Hit, application found, was dpd10+ last 3 months → finstar_hit_app_found_dpd10

Decisioning today:
	•	MRB=0: we do not reject customers based on these Finstar-hit rules.
	•	MRB=1: we reject customers based on these Finstar-hit rules.

Applies to all segments:
	•	CLN → CLN
	•	POS → CLN
	•	Xsell → CLN
	•	COST → CLN (same logic; metrics to be confirmed if needed)

⸻

Expected Behavior

Implement the following segment-agnostic logic (apply to CLN→CLN, POS→CLN, Xsell→CLN, COST→CLN):

1) MRB=0 (tighten — start rejecting)
	•	Reject / block disbursements for:
	•	finstar_hit_app_found_last_reject
	•	finstar_hit_app_found_dpd10

2) MRB=1 (relax — allow one bucket)
	•	Do NOT reject for:
	•	finstar_hit_app_not_found  (open for lending / allow issuance)

3) Keep as-is (no change unless specified)
	•	finstar_hit_app_found remains under current MRB behavior (i.e., still restricted in MRB=1 unless you explicitly want to change it).

⸻

If you want, I can add a short Acceptance Criteria block (what exact checks Scoi/Team should validate in decision logs for each rule + MRB + segment).