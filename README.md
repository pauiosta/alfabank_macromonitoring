Hi Yen,

Here are my bank's USD routing instructions:

Beneficiary: Tazin Pavel Sergeevich
Account number: 1242020006354820
Currency: USD

Beneficiary Bank: OJSC BAKAI BANK
Location: Bishkek, Kyrgyz Republic
SWIFT: BAKAKG22
Account: TR280014300000000009443581

Correspondent Bank: Aktif Yatirim Bankasi A.S.
Location: Istanbul, Turkey
SWIFT: CAYTTRIS

Intermediary Bank: Raiffeisen Bank International AG
Location: Vienna, Austria
SWIFT: RZBAATWW

Please resend the 3 payments using these routing details.

Thank you.
Pavel

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