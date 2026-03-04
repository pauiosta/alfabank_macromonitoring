Here are the three texts updated with the precise rule definitions:

Business Value
This change is expected to improve overall portfolio profitability by refining the segmentation logic within the Pwinstar rule group. By opening the Finstar Hit — Application Not Found segment at MRB=1, we are targeting a population whose observed risk profile is comparable to the currently approved baseline (finstar_no_hit), making this a justified expansion of the credit-granting perimeter.
At the same time, maintaining the reject decision for Finstar Last Reject and Finstar DPD 10+ clients preserves portfolio quality, as these segments demonstrate materially elevated early delinquency and strongly negative NPV that do not support profitable lending under current model assumptions.
Segments being opened — Finstar Hit, Application Not Found (MRB=1):



|Segment    |FPD7+|D4P6 |Avg NPV|Predicted NPV (MRB=1)|vs. Baseline                          |
|-----------|-----|-----|-------|---------------------|--------------------------------------|
|CLN → CLN  |4.1% |11.5%|1,934  |—                    |Comparable to baseline (5.2% / 1,966) |
|POS → CLN  |6.8% |11.5%|1,021  |334 (D4P6: 16.15%)   |Acceptable vs. baseline (5.3% / 1,344)|
|Xsell → CLN|3.9% |8.8% |1,335  |473 (D4P6: 14.05%)   |Comparable to baseline (3.7% / 1,818) |

Segments being kept as reject — Last Reject & DPD 10+:



|Segment              |FPD7+|D4P6 |Avg NPV|Issue                                     |
|---------------------|-----|-----|-------|------------------------------------------|
|CLN→CLN Last Reject  |10.7%|22.4%|-106   |2× baseline FPD7+, negative NPV           |
|CLN→CLN DPD10+       |17.8%|34.3%|-1,350 |3× baseline FPD7+, strongly negative NPV  |
|POS→CLN Last Reject  |14.1%|21.1%|-144   |2.7× baseline FPD7+, negative NPV         |
|POS→CLN DPD10+       |13.1%|23.8%|-811   |2.5× baseline FPD7+, negative NPV         |
|Xsell→CLN Last Reject|8.1% |16.0%|-1,091 |2.2× baseline FPD7+, negative NPV         |
|Xsell→CLN DPD10+     |10.2%|17.7%|-2,384 |2.8× baseline FPD7+, strongly negative NPV|

Overall, this change is projected to increase portfolio profitability and reduce the rate of early delinquency across all three segments:
	∙	CLN → CLN: risk reduced from 12.2% → 10.5%, Avg NPV increased from 1,024 → 1,625
	∙	POS → CLN: risk reduced from 11.4% → 10.7%, Avg NPV increased from 810 → 999
	∙	Xsell → CLN: risk reduced from 8.9% → 8.3%, Avg NPV increased from 1,865 → 2,214

Current Behavior
The Pwinstar rule group currently contains four rules, all of which result in a reject decision at the maximum of MRB0 for any applicant who triggers them:
	1.	finstar_hit_app_found — Finstar Hit: application was found. The applicant is present in the Finstar database and a matching application record was found.
	2.	finstar_hit_app_not_found — Finstar Hit: application was not found. The applicant is present in the Finstar database, but no matching application record was found.
	3.	finstar_hit_app_found_last_reject — Finstar Hit: application found, no agreement, last reject was within the past 1 year.
	4.	finstar_hit_app_found_dpd10 — Finstar Hit: application found, had DPD10+ within the last 3 months.
All four rules apply uniformly across all product segments (CLN→CLN, POS→CLN, Xsell→CLN): any client matching any of these conditions is declined, with the offer capped at the maximum of MRB0.

Expected Behavior
Following this change, the four Pwinstar rules will be treated differently based on the demonstrated risk profile of each underlying subsegment.
finstar_hit_app_found_last_reject (application found, no agreement, last reject ≤ 1 year ago) and finstar_hit_app_found_dpd10 (application found, had DPD10+ in the last 3 months) will continue to result in a reject decision, capped at the maximum of MRB0. Across all segments, these populations show FPD7+ rates of 8–18% and strongly negative average NPVs, indicating a concentration of risk that is not captured by the PD model alone and cannot be profitably served at this time.
finstar_hit_app_found (application found) remains unchanged and continues to result in a reject decision at the maximum of MRB0.
finstar_hit_app_not_found (application was not found) will no longer trigger a reject. Clients matching this rule will instead be eligible for approval with the offer set at the maximum of MRB1. The rationale per segment is as follows:
	∙	CLN → CLN: FPD7+ of 4.1% and Avg NPV of 1,934 are comparable to the baseline (5.2% / 1,966). The segment does not differ significantly from non-Finstar clients.
	∙	POS → CLN: FPD7+ of 6.8% and Avg NPV of 1,021 are moderately above baseline but acceptable. At MRB=1, the predicted D4P6 is 16.15% with a positive NPV of 334, supporting disbursement. In contrast, finstar_hit_app_found at MRB=1 yields a predicted NPV of -882, confirming that only the app_not_found subsegment should be opened.
	∙	Xsell → CLN: FPD7+ of 3.9% and Avg NPV of 1,335 are closely in line with the baseline (3.7% / 1,818). At MRB=1, the predicted D4P6 is 14.05% with a positive NPV of 473, confirming that finstar_hit_app_not_found MRB=1 can be opened for lending.​​​​​​​​​​​​​​​​



Понял, спасибо за уточнение. Переписываю:

Current Behavior
The Pwinstar rule group currently contains four rules that apply exclusively to clients in the MRB1 bucket. Any applicant in MRB1 who triggers any of these rules receives a reject decision. Clients in MRB0 are not affected by these rules.
The four rules are:
	1.	finstar_hit_app_found — Finstar Hit: application was found. The applicant is present in the Finstar database and a matching application record was found.
	2.	finstar_hit_app_not_found — Finstar Hit: application was not found. The applicant is present in the Finstar database, but no matching application record was found.
	3.	finstar_hit_app_found_last_reject — Finstar Hit: application found, no agreement, last reject was within the past 1 year.
	4.	finstar_hit_app_found_dpd10 — Finstar Hit: application found, had DPD10+ within the last 3 months.
All four rules result in a reject for MRB1 clients across all product segments (CLN→CLN, POS→CLN, Xsell→CLN).

Expected Behavior
Following this change, the four Pwinstar rules within the MRB1 bucket will be treated differently based on the demonstrated risk profile of each subsegment.
finstar_hit_app_found_last_reject (last reject ≤ 1 year ago) and finstar_hit_app_found_dpd10 (DPD10+ in the last 3 months) will continue to result in a reject for MRB1 clients. Across all segments these populations show FPD7+ rates of 8–18% and strongly negative average NPVs, indicating a concentration of risk that is not captured by the PD model alone and cannot be profitably served at this time.
finstar_hit_app_found (application found) remains unchanged and continues to result in a reject for MRB1 clients.
finstar_hit_app_not_found (application was not found) will no longer trigger a reject in MRB1. Clients matching this rule will now be approved at MRB1. The rationale per segment:
	∙	CLN → CLN: FPD7+ of 4.1% and Avg NPV of 1,934 are comparable to the baseline (5.2% / 1,966), confirming the segment does not differ significantly from non-Finstar clients.
	∙	POS → CLN: At MRB=1, predicted D4P6 is 16.15% with a positive NPV of 334, supporting disbursement. In contrast, finstar_hit_app_found at MRB=1 yields a predicted NPV of -882, confirming only the app_not_found subsegment should be opened.
	∙	Xsell → CLN: At MRB=1, predicted D4P6 is 14.05% with a positive NPV of 473, confirming that finstar_hit_app_not_found MRB=1 can be opened for lending.
MRB0 clients remain unaffected by this change.

Business Value
(секцию Business Value менять не нужно — она остаётся актуальной, только уточни что речь идёт именно о MRB1 сегменте) — хочешь, перепишу и её с этим уточнением?​​​​​​​​​​​​​​​​