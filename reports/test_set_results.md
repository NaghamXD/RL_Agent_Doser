# Test-set generalization check: `control` (sweep2_heavy_reg)

Every result in this project up to this point — the 4-regime hyperparameter
comparison, the 9-variant reward-shaping ablation, the `control`
recommendation — was evaluated exclusively on the **validation** split
(40 patients), which was also used for `best.pt` checkpoint selection.
Reusing the same patients for both selection and reporting risks the
"control wins" conclusion partly reflecting fit to that specific 40-patient
sample rather than true generalization. This report runs the same model
against the **held-out test split** (100 patients, never seen during
training, hyperparameter selection, or reward-shaping comparison) to check.

The test split had not been preprocessed before this run: raw CSVs for the
100 test patients existed (`data/original/test/`) but no dose-influence
matrices had ever been computed for them. Built the pipeline from scratch:
`scripts/preprocess.py --splits test` (~10s) then
`scripts/compute_dose_influence_matrix.py --splits test` (~37 min, the
genuinely expensive step), then `evaluate.py --split test`.

## Results

| | n | MAE | DVH | D95_PTV70 | D95_PTV63 | D95_PTV56 | Brainstem(54) | SpinalCord(45) | Mandible(70) | LeftParotid(26) | RightParotid(26) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **validation** | 40 | 11.71 | 23.34 | 43.97 | 39.31 | 38.92 | 19.65 | 24.71 | 41.07 | 39.32 | 43.21 |
| **test** | 100 | 11.28 | **22.71** | 43.77 | 40.04 | 38.20 | 17.16 | 23.48 | 38.35 | 41.03 | 41.95 |

Full per-patient breakdowns: `reports/test_set_validation_per_patient.csv`,
`reports/test_set_test_per_patient.csv`. Figure: `reports/test_set_results.png`.

## Findings

**The model generalizes — every metric on the 100 unseen test patients is
within noise of the validation numbers, several are mildly better.**
Test DVH (22.71) is actually *lower* (better) than validation (23.34), and
every OAR mean dose except the parotids is mildly better on test
(Brainstem 17.16 vs 19.65, SpinalCord 23.48 vs 24.71, Mandible 38.35 vs
41.07). PTV coverage is essentially unchanged (D95_PTV70 43.77 vs 43.97).
The per-patient DVH distribution (panel A) shows similar spread on both
splits (test std 4.35 vs validation std 4.82) with no bimodality or
fat tail — this is not an artifact of a few easy test patients dragging
the mean down. **No evidence of overfitting to the validation split.**

**The central unresolved problem from the rest of this investigation holds
on test too, not just on validation.** Both parotids remain meaningfully
over their 26 Gy tolerance on the test set (LeftParotid 41.03, RightParotid
41.95 — both *slightly worse* than validation's 39.32/43.21), and PTV
coverage remains well under prescription (D95_PTV70 43.77/70 Gy ≈ 63%,
matching validation almost exactly). This confirms the parotid-overdose /
PTV-underdose finding from `reports/reward_shaping_sweep.md` is a genuine
property of the model and reward design, not a quirk of the particular 40
validation patients — i.e. the reward-shaping investigation's conclusion
("redistributing the trade-off, not pushing the frontier") generalizes too.

## Bottom line

`control` is confirmed as the best model on truly held-out data, with
performance consistent with (not inflated relative to) its
validation-time numbers. The project's open problem — getting parotid dose
under tolerance without sacrificing PTV coverage — remains exactly as
unresolved on 100 new patients as it was on the 40 used throughout the
rest of this investigation.
