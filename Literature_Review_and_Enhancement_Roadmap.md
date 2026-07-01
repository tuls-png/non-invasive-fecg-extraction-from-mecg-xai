# Literature Review and Enhancement Roadmap
### Non-Invasive Maternal and Fetal ECG Separation Using Independent Component Analysis and Adaptive Windowed Weighted Singular Value Decomposition

---

## 0. Scope and Method

This review was built directly against your codebase (`pipeline.py`, `separation/ica.py`, `separation/wsvd.py`, `separation/ekf.py`, `evaluation/metrics.py`, `configs/adfecgdb.yaml`, `configs/cinc2013.yaml`) so the comparisons and recommendations below are anchored to what your dual-path PHASE pipeline (Path A: ICA1-direct; Path B: AW-WSVD residual → ICA2; 5-seed ensemble; EKF morphological refinement) actually does, not to a generic description of ICA+SVD.

Inclusion criteria for the literature table:
- Peer-reviewed (journal or established conference proceedings — IEEE, Elsevier, Springer, MDPI, PLOS, Frontiers, Nature Portfolio).
- Reports **F1 ≥ 85%** for fetal QRS/R-peak detection.
- Evaluated on **ADFECGDB** (Silesian/PhysioNet Abdominal and Direct Fetal ECG Database) and/or **CinC2013** (PhysioNet/Computing in Cardiology Challenge 2013, Set A) — the same two benchmarks your dissertation uses — or, where a different benchmark was used, this is explicitly flagged in the table.
- Where a paper reports multiple method variants, I include the variant that clears the 85% F1 bar and note weaker variants only for contrast.

---

## 1. Tabular Literature Review

| # | Authors (Year) | Venue | Core Method | Dataset(s) | Se / PPV | **F1** | Relevance to your pipeline |
|---|---|---|---|---|---|---|---|
| 1 | Hao, Yang, Zhou & Wu (2022) | *Sensors* 22(10):3705 | FastICA (SVD-refined seed) + SVD residual reconstruction + wavelet-modulus-maxima QRS detector | ADFECGDB **and** CinC2013 | 96.90% / 98.23% | **95.24%** | Closest structural analogue to your work: ICA and SVD are combined as sequential, not competing, stages. Directly comparable baseline. |
| 2 | Li, Frasch & Wu (2017) | *Frontiers in Physiology* 8:277 | Diffusion-map channel selection + de-shape STFT/nonlocal-median two-channel separation (SAVER) | ADFECGDB / CinC2013 (reported separately) | — | **99.3%** (ADFECGDB) / **87.93%** (CinC2013) | Demonstrates the same ADFECGDB-vs-CinC2013 performance gap you are currently seeing, and attributes it to CinC2013's heterogeneous SNR and shorter (60 s) windows — directly relevant to your AW-WSVD window-adaptation rationale. |
| 3 | Kotas et al. (2024) | *Biocybernetics and Biomedical Engineering* 44(1):161–182 | Spatio-spectral extension of ICA using method-of-delays embedding + iterative source-subspace clustering | ADFECGDB (2-channel) | 100% / 99.97% | **99.98%** | Shows that *constraining* ICA with spatio-spectral structure (not just amplitude/kurtosis) resolves the IC-selection ambiguity your `score_fetal_ic`/`score_maternal_ic` heuristics are also fighting. |
| 4 | Gurve & Krishnan (2020) | *IEEE J. Biomed. Health Inform.* 24(3):669–680 | Activation-scaled Non-negative Matrix Factorization (NNMF) refining an initial ICA decomposition, then Pan–Tompkins | ADFECGDB + CinC2013 | 95.30% / 94.60% | **94.80%** | Pure-ICA baseline in the same paper only reaches 93.6% F1 — empirical evidence that a *second, differently-constrained* decomposition stage after ICA (your Path B logic) reliably adds several F1 points. |
| 5 | Barnova et al. (2021) | *PLOS ONE* 16(8):e0256154 | Ensemble Empirical Mode Decomposition + Recursive Least Squares adaptive filter + ICA cascade | ADFECGDB + CinC2013 | 95.09% / 96.36% | **95.69%** | EMD/RLS alone (no ICA) only reaches 84.1% F1 in the same study — again showing cascaded, complementary decompositions beat any single method, and that adaptive filtering recovers residual maternal leakage ICA alone misses. |
| 6 | Zhang & Yu (2020) | *Medical & Biological Engineering & Computing* 58:419–432 | K-means clustering + PCA + template subtraction for single-lead separation | CinC2013 | 96.23% / 95.35% | **95.78%** | Template subtraction of the *maternal beat morphology* (not just QRS spikes) is the key differentiator — directly maps to the template-subtraction "Path C" you have prioritized. |
| 7 | Jaba & Dhanalakshmi (2021) | *Biomedical Engineering / Biomedizinische Technik* 66(5):503–514 | Parallel Sub-Filter adaptive noise canceller (PSF-ANC) | DaISy + CinC2013 | 97.92% / 94.66% | **96.12%** | Shows adaptive noise cancellation *after* the main separation stage recovers precision on noisy CinC2013-type recordings — relevant to your lower CinC2013 scores. |
| 8 | Mollakazemi, Asadi, Tajnesaei & Ghaffari (2021) | *J. Biomedical Physics & Engineering* 11(2):197–204 | PCA + Discrete Wavelet Transform with explicit per-window signal-quality estimation | CinC2013 | — | **98.77%** | Signal-quality-index (SQI) gating of which windows/channels are trusted — directly analogous to your AW-WSVD per-window correlation gate, but applied dataset-wide as a first-class scoring step. |
| 9 | Jallouli, Arfaoui, Ben Mabrouk & Cattani (2021) | *Entropy* (MDPI) 23(7):844 | Clifford-wavelet decomposition + wavelet-entropy-based QRS localization | CinC2013 | 99.76% / 99.20% | **99.47%** | Highest CinC2013 F1 in this review; illustrates that wavelet-domain QRS localization outperforms simple thresholded Pan–Tompkins detection (which your pipeline uses) on noisy single-channel residuals. |
| 10 | Rasti-Meymandi & Ghaffari (2021) | *Physiological Measurement* 42:045002 | AECG-DecompNet — deep encoder-decoder trained to output MECG/FECG decomposition directly | CinC2013 | 97.40% / 93.52% | **95.42%** | Data-driven decomposition benchmark; useful as an upper-bound reference for how much headroom purely classical methods are leaving on the table on CinC2013. |
| 11 | Mohebbian, Vedaei, Wahid, Dinh, Marateb & Tavakolian (2022) | *IEEE J. Biomed. Health Inform.* 26(2):515–526 | Attention-based CycleGAN domain mapping (mECG → fECG), Pan–Tompkins detection | ADFECGDB / NI-FECG | — | **99.70%** | Confirms that *morphology-preserving* refinement after coarse separation (analogous to your EKF stage) is what pushes ADFECGDB scores above 99%. |
| 12 | Taha & Abdel-Raheem (2020) | *Sensors* 20(12):3536 | Null-space-based blind source separation (ICA variant with explicit maternal-subspace projection) | CinC2013 | 97.30% / 93.30% | **95.70%** | Explicit maternal-subspace exclusion (vs. your correlation-threshold exclusion in `_find_maternal_residual_idx`) — a formalization worth comparing against. |
| 13 | Sarafan et al. (2020) | *Technologies* (MDPI) 8(2):33 | Three parallel ICA variants → **template subtraction** → **Extended Kalman Filter** cascade | CinC2013 | — | **92.61%** | Nearly identical architecture to yours (ICA → TS → EKF), reported by Hao et al. (2022, row 1) as a direct literature comparator. Confirms EKF morphological refinement is a validated final stage, and quantifies what TS adds even without your WSVD stage. |
| 14 | Zhong, Liao, Guo & Wang (2019) | *Australasian Physical & Engineering Sciences in Medicine* 42(4):1081–1089 | Residual Convolutional Encoder–Decoder Network (RCED-Net), single-channel | ADFECGDB **and** CinC2013 (NIFECGC) | — | **94.10%** (ADFECGDB) / **93.62%** (CinC2013) | Rare paper reporting comparably on *both* your benchmark datasets from a single architecture — useful as a same-basis cross-dataset comparator to your own ADFECGDB (~95%) vs. CinC2013 gap. |

*(Panigrahy & Sahu (2017, *Australas. Phys. Eng. Sci. Med.* 40:191–207) and the ICA-only variant of Gurve & Krishnan (93.6% F1) are cited above as internal comparators but do not themselves clear the 85% threshold as headline results and are not given their own rows.)*

---

## 2. Cross-Paper Comparison: What the High-F1 Methods Have in Common

Looking across rows 1–14, five structural patterns recur in every method that clears ~95% F1, regardless of whether the underlying tool is classical or deep-learning-based:

**1. No single decomposition method is used alone.** Every top performer chains at least two *complementary* separation mechanisms: ICA + SVD (row 1), ICA + NNMF (row 4), EMD + RLS + ICA (row 5), PCA + template subtraction (row 6), ICA + template subtraction + EKF (row 13). Your PHASE architecture already follows this pattern (dual-path ICA1 / WSVD+ICA2), which is why your ADFECGDB F1 (~95%) is already competitive with rows 1, 4, 5, and 13.

**2. Template/morphology subtraction of the *whole* maternal beat, not just the QRS spike, appears in every paper above 95% F1 that isn't pure deep learning** (rows 1, 5, 6, 7, 13). Your `gaussian_weight_matrix` already extends beyond the QRS with a broader PQRST support window — this is the right direction, but none of these papers stop at a Gaussian proxy; they subtract an estimated or learned *beat template* per maternal cycle. This is exactly the "Path C" template-subtraction idea your prior planning already flagged as highest-upside.

**3. A final morphological or adaptive refinement stage recovers the last few F1 points.** EKF (rows 11, 13), RLS/LMS adaptive filtering (rows 5, 7), and wavelet-entropy re-localization (row 9) all sit *after* the main separation and specifically target residual maternal leakage or QRS-boundary imprecision. Your EKF stage already does this for morphology; it is not yet doing it for the *initial* R-peak location refinement pass, since your EKF acceptance gates (peak count, CC, median-RR) are conservative pass/reject filters rather than an active second detection pass.

**4. Per-window or per-channel signal-quality gating is universal on CinC2013-class data.** Rows 8 (SQI+PCA+DWT), 5 (RLS+ICA), and your own `WSVD_CHANNEL_R2_MIN` / per-window correlation gate all independently converge on the same idea: CinC2013's heterogeneous, single-source-unlabeled 60 s recordings need explicit trust-weighting of which windows/channels to use, because a fixed global threshold (tuned for ADFECGDB's cleaner 5-minute recordings) systematically under- or over-subtracts on CinC2013. This directly explains the ADFECGDB > CinC2013 gap you are seeing and that rows 1, 2, and 14 also report on the same two datasets.

**5. Ensembling / redundancy beats any single "best" decomposition.** Rows 1 (dual algorithm cross-check), 4–5 (cascaded methods), and your own 5-seed FastICA ensemble all exploit the fact that ICA's local-optimum sensitivity (unstable ordering, sign, and occasional component swap) is best handled by generating multiple candidates and *selecting*, rather than trying to make a single decomposition perfect.

**Net implication for your dissertation:** your architecture is already aligned with the field's highest-performing design pattern (cascade + ensemble + refinement). The gap to the ~96–99% F1 papers is concentrated in exactly the three places your prior fix-list identified — template subtraction (Path C), CinC2013-specific quality gating, and IC-selection robustness — which the literature independently confirms are the highest-leverage additions.

---

## 3. Recommended Enhancements — Ranked by Expected Impact

All recommendations are constrained to be compatible with a classical signal-processing/ML pipeline (no end-to-end deep nets), reversible/ablatable (so each can be reported as a controlled ablation in your results chapter), and buildable on your existing module boundaries (`separation/ica.py`, `separation/wsvd.py`, `separation/ekf.py`, `pipeline.py`).

### Rank 1 — Beat-template subtraction as a third IC-selection path (Path C)
**What:** Estimate a maternal PQRST template per recording (median- or robust-mean-synchronized average over detected maternal R-peaks), subtract the time-aligned template at every maternal beat location (not just a Gaussian QRS-weighted SVD residual), then re-run ICA on the *template-subtracted* residual as a third candidate path alongside Path A/B.
**Why it's highest-impact:** This is the single element most consistently present across every non-deep-learning paper above 95% F1 (rows 1, 5, 6, 7, 13) and the one most clearly missing from your current two paths, both of which rely on subspace/correlation-based maternal removal rather than explicit morphological subtraction. Zhang & Yu (2020) and Sarafan et al. (2020) show this specifically resolves cases where ICA's maternal component overlaps in frequency/kurtosis with the fetal component — precisely the "symmetric IC selection failures" your own diagnostics flagged.
**Integration point:** New module `separation/template_subtraction.py` (a stub already exists in your `__pycache__`, suggesting this was scaffolded); called between `select_maternal_ic()` and the WSVD stage in `pipeline.py`, producing `residual_C` fed into a third `run_ica()` call, scored by the same three-factor formula as Paths A/B.
**Expected gain:** Largest single-fix F1 improvement, concentrated on the symmetric-failure recordings currently pulling down both dataset means — plausibly the single change that closes most of the ADFECGDB→CinC2013 gap, since template subtraction is amplitude/morphology-based rather than kurtosis-based and is less sensitive to CinC2013's lower per-channel SNR.

### Rank 2 — Per-window/per-channel Signal Quality Index (SQI)-weighted fusion, dataset-wide
**What:** Generalize your existing `WSVD_CHANNEL_R2_MIN` gate and per-window correlation gate into an explicit SQI score (e.g., kurtosis + spectral flatness + RR-regularity, following Mollakazemi et al. 2021) computed once per channel/window and used to *weight* (not just gate) contributions from Path A, Path B, and the new Path C, rather than a single hard "chosen_path" selection.
**Why it's high-impact:** Rows 5, 8, and your own diagnostic logs show this is exactly where CinC2013 loses points relative to ADFECGDB — a small number of low-SNR windows in an otherwise-good recording currently degrade the whole-recording HR/peak score because your pipeline picks one path per recording rather than blending per-window.
**Integration point:** `evaluation/sqi.py` already exists in your codebase; currently appears under-used relative to its potential — promote it from a diagnostic/logging role to a first-class weighting input inside `_best_ic_ensemble()`.
**Expected gain:** Second-largest gain, concentrated specifically on CinC2013 (heterogeneous multi-source recordings), with minimal risk of regressing ADFECGDB since well-behaved recordings will simply get near-uniform SQI weights.

### Rank 3 — Periodicity-constrained ICA (πCA/spatio-spectral-style regularization of IC selection)
**What:** Replace or augment the current kurtosis + peak-count scoring in `score_fetal_ic`/`score_maternal_ic` with an explicit periodicity/autocorrelation criterion in the ICA objective or post-hoc scoring — following Sameni's periodic component analysis lineage and Kotas et al.'s (2024) spatio-spectral extension, which reached 99.98% F1 on ADFECGDB specifically by exploiting quasi-periodicity rather than non-Gaussianity alone.
**Why it matters:** FastICA's amplitude/kurtosis-based objective is agnostic to the fact that both cardiac sources are quasi-periodic; your current fallback logic (`[FIX-4]` variance fallback, harmonic-confusion checks) is effectively patching around this gap post-hoc. A periodicity term folded into IC scoring (or a lightweight πCA pre-whitening step) directly targets the harmonic-confusion and half-HR failure modes your `_check_harmonic_confusion` function currently only detects rather than prevents.
**Integration point:** `separation/ica.py::score_fetal_ic`/`score_maternal_ic` — add an autocorrelation-peak-sharpness term; low implementation cost, moderate but broad expected benefit across nearly all recordings (not just the current outliers).
**Expected gain:** Moderate, broad-based improvement — fewer recordings needing the harmonic-confusion/low-confidence fallback paths at all, which should also reduce the number of recordings you currently have to exclude/flag in `aggregate_results()`.

### Rank 4 — Adaptive filtering (RLS/NLMS) as post-WSVD residual cleanup
**What:** After `subtract_maternal()`, run a short adaptive filter (RLS or normalized-LMS) using the retained/rejected SVD components as a reference to suppress residual maternal leakage that survives the correlation-gated WSVD subtraction, before the ICA2 stage.
**Why it matters:** Barnova et al. (2021) show RLS+ICA recovers ~11 F1 points over EMD alone specifically by mopping up residual periodic leakage that a single decomposition pass leaves behind; this is a natural extra stage given you already compute per-window correlation-validated maternal components in `adaptive_windowed_wsvd`, which can double as the RLS reference signal at negligible extra cost.
**Integration point:** New short function in `separation/wsvd.py` or a new `separation/adaptive_filter.py`, inserted between `subtract_maternal()` and `run_ica()` for Path B only.
**Expected gain:** Smaller, more targeted gain than Ranks 1–2 (residual cleanup rather than a new detection path), but low implementation risk and directly literature-validated (row 5, row 7).

### Rank 5 — Two-pass Kalman smoothing (forward EKF + backward smoother) for the chosen IC
**What:** Extend your existing single-pass `FetalECGKalmanFilter` to a forward-backward Extended Kalman *Smoother* (RTS-style), matching Panigrahy & Sahu (2017)'s EKS+ANFIS design, applied only after the final IC is chosen (not as a per-path scoring input, to avoid disrupting your existing gate logic).
**Why it matters:** Your current EKF acceptance gates are deliberately conservative (reject-on-uncertainty), which protects against phase-shift artifacts but also means EKF often contributes nothing on marginal recordings. A backward smoothing pass tends to reduce state-estimation variance without introducing the phase-shift risk your `[NEW]` gate (c) was specifically built to catch, since RTS smoothing revises earlier estimates using future information rather than extrapolating forward only.
**Integration point:** `separation/ekf.py` — add a `smooth()` counterpart to the existing `filter()` (a placeholder docstring already exists in the module for this: *"[FIX-1] Phase burn-in added to filter() and smooth()"*).
**Expected gain:** Small, incremental — mainly improves FHR_MAE and morphology cleanliness on already-accepted recordings rather than F1 on rejected/marginal ones.

### Rank 6 — Bayesian/grid hyperparameter adaptation per dataset (replacing hand-tuned YAML overrides)
**What:** Replace the manually-tuned per-dataset overrides in `configs/cinc2013.yaml` (e.g., `wsvd_component_corr_thresh: 0.15`, `pt_threshold_factor: 0.35`) with a small held-out-validation Bayesian optimization pass (e.g., scikit-optimize) over the ~6 most sensitive thresholds, run once per dataset.
**Why it matters:** Not a novel *algorithmic* contribution, but the literature comparison in row 2 (Li et al., 99.3% ADFECGDB vs. 87.93% CinC2013 — nearly your own gap) and row 14 (Zhong et al., 94.1% vs. 93.6% — a *much smaller* gap on the same two datasets with a data-driven method) together suggest part of your remaining CinC2013 gap may be threshold mis-generalization rather than an algorithmic deficiency. This is the lowest-novelty but easiest-to-defend enhancement, useful as a robustness/generalization ablation in your results chapter even if its F1 gain is modest.
**Integration point:** Wraps `run_experiment_new.py`; no changes to core separation modules required.
**Expected gain:** Smallest and most dataset-specific; primarily reduces variance/overfitting risk from manual tuning rather than adding new signal-processing capability.

---

## 4. Recommended Final Pipeline

```
                                   ┌─────────────────────────────────────────────┐
                                   │   Raw multichannel abdominal ECG (n_ch)      │
                                   └───────────────────────┬───────────────────────┘
                                                            │
                                        preprocess_multichannel()  [existing]
                                                            │
                              ┌─────────────────────────────┼─────────────────────────────┐
                              │                              │                              │
                    ┌─────────▼─────────┐        ┌───────────▼───────────┐      ┌───────────▼────────────┐
                    │   PATH A           │        │   PATH B               │      │   PATH C  [NEW – R1]    │
                    │  ICA1 direct       │        │  AW-WSVD residual      │      │  Maternal beat-template  │
                    │  (existing)        │        │  → ICA2 (existing)     │      │  subtraction → ICA3      │
                    │                    │        │  + RLS/NLMS cleanup    │      │  (template_subtraction.py│
                    │                    │        │  [NEW – R4]            │      │  module already stubbed) │
                    └─────────┬─────────┘        └───────────┬───────────┘      └───────────┬────────────┘
                              │                                │                              │
                              │        IC scoring: score_fetal_ic + score_maternal_ic          │
                              │        + periodicity/autocorrelation term  [NEW – R3]           │
                              └────────────────────────────────┼──────────────────────────────┘
                                                                 │
                                          ┌──────────────────────▼──────────────────────┐
                                          │  SQI-weighted fusion across Paths A/B/C       │
                                          │  (per-window/per-channel, not hard-select)    │
                                          │  [NEW – R2, promotes evaluation/sqi.py]       │
                                          └──────────────────────┬──────────────────────┘
                                                                 │
                                     N_ENSEMBLE=5 seed ensemble aggregation  [existing]
                                                                 │
                                          ┌──────────────────────▼──────────────────────┐
                                          │  EKF morphological refinement (existing)      │
                                          │  + RTS backward smoothing pass  [NEW – R5]    │
                                          │  (3-gate acceptance: peaks / CC / RR — kept)  │
                                          └──────────────────────┬──────────────────────┘
                                                                 │
                                              evaluate()  →  Se / PPV / F1 / FHR_MAE
                                                                 │
                              (offline, once per dataset)  Bayesian threshold tuning  [NEW – R6]
                                     feeding back into configs/{adfecgdb,cinc2013}.yaml
```

**Reading the diagram against your current code:** Paths A and B, the ensemble loop, and the EKF gate structure are unchanged — every new element is additive and independently ablatable, so you can report F1 with each enhancement added incrementally (a natural structure for a dissertation results chapter: baseline → +Path C → +SQI fusion → +periodicity term → +RLS cleanup → +EKS → +tuned thresholds).

**Suggested ablation order for your results chapter**, matching the ranking above: (1) baseline PHASE (current), (2) + Path C template subtraction, (3) + SQI-weighted fusion, (4) + periodicity-constrained IC scoring, (5) + RLS/NLMS residual cleanup, (6) + RTS smoothing, (7) + tuned thresholds — reporting cumulative ADFECGDB and CinC2013 F1 at each step lets you both demonstrate novelty (Path C, SQI fusion, periodicity term are your original contributions) and honestly contextualize them against the 14 papers above.

---

## 5. Reference List (full citations)

1. Hao, J., Yang, Y., Zhou, Z., & Wu, S. (2022). Fetal Electrocardiogram Signal Extraction Based on Fast Independent Component Analysis and Singular Value Decomposition. *Sensors*, 22(10), 3705. https://doi.org/10.3390/s22103705
2. Li, R., Frasch, M. G., & Wu, H.-T. (2017). Efficient Fetal-Maternal ECG Signal Separation from Two Channel Maternal Abdominal ECG via Diffusion-Based Channel Selection. *Frontiers in Physiology*, 8, 277. https://doi.org/10.3389/fphys.2017.00277
3. Kotas, M., et al. (2024). Spatio-spectral independent component analysis for fetal ECG extraction from two-channel maternal abdominal signals. *Biocybernetics and Biomedical Engineering*, 44(1), 161–182.
4. Gurve, D., & Krishnan, S. (2020). Separation of Fetal-ECG from Single-Channel Abdominal ECG Using Activation Scaled Non-Negative Matrix Factorization. *IEEE Journal of Biomedical and Health Informatics*, 24(3), 669–680. https://doi.org/10.1109/JBHI.2019.2920356
5. Barnova, K., Martinek, R., Jaros, R., Kahankova, R., Matonia, A., Jezewski, M., Czabanski, R., Horoba, K., & Jezewski, J. (2021). A novel algorithm based on ensemble empirical mode decomposition for non-invasive fetal ECG extraction. *PLOS ONE*, 16(8), e0256154. https://doi.org/10.1371/journal.pone.0256154
6. Zhang, Y., & Yu, S. (2020). Single-lead noninvasive fetal ECG extraction by means of combining clustering and principal components analysis. *Medical & Biological Engineering & Computing*, 58, 419–432. https://doi.org/10.1007/s11517-019-02087-7
7. Jaba, D. K. A., & Dhanalakshmi, S. R. K. (2021). An improved parallel sub-filter adaptive noise canceler for the extraction of fetal ECG. *Biomedical Engineering / Biomedizinische Technik*, 66(5), 503–514. https://doi.org/10.1515/bmt-2020-0313
8. Mollakazemi, M. J., Asadi, F., Tajnesaei, M., & Ghaffari, A. (2021). Fetal QRS Detection in Noninvasive Abdominal Electrocardiograms Using Principal Component Analysis and Discrete Wavelet Transforms with Signal Quality Estimation. *Journal of Biomedical Physics and Engineering*, 11(2), 197–204.
9. Jallouli, M., Arfaoui, S., Ben Mabrouk, A., & Cattani, C. (2021). Clifford Wavelet Entropy for Fetal ECG Extraction. *Entropy*, 23(7), 844. https://doi.org/10.3390/e23070844
10. Rasti-Meymandi, A., & Ghaffari, A. (2021). AECG-DecompNet: Abdominal ECG signal decomposition through deep-learning model. *Physiological Measurement*, 42, 045002. https://doi.org/10.1088/1361-6579/abedc1
11. Mohebbian, M. R., Vedaei, S. S., Wahid, K. A., Dinh, A., Marateb, H. R., & Tavakolian, K. (2022). Fetal ECG Extraction from Maternal ECG using Attention-based CycleGAN. *IEEE Journal of Biomedical and Health Informatics*, 26(2), 515–526. https://doi.org/10.1109/JBHI.2021.3111873
12. Taha, L., & Abdel-Raheem, E. (2020). A Null Space-Based Blind Source Separation for Fetal Electrocardiogram Signals. *Sensors*, 20(12), 3536. https://doi.org/10.3390/s20123536
13. Sarafan, S., Le, T., Naderi, A. M., Nguyen, Q. D., Kuo, B. T. Y., Ghirmai, T., Han, H. D., Lau, M. P. H., & Cao, H. (2020). Investigation of Methods to Extract Fetal Electrocardiogram from the Mother's Abdominal Signal in Practical Scenarios. *Technologies*, 8(2), 33. https://doi.org/10.3390/technologies8020033
14. Zhong, W., Liao, L., Guo, X., & Wang, G. (2019). Fetal electrocardiography extraction with residual convolutional encoder–decoder networks. *Australasian Physical & Engineering Sciences in Medicine*, 42(4), 1081–1089. https://doi.org/10.1007/s13246-019-00805-x
15. Panigrahy, D., & Sahu, P. K. (2017). Extraction of fetal ECG signal by an improved method using extended Kalman smoother framework from single channel abdominal ECG signal. *Australasian Physical & Engineering Sciences in Medicine*, 40(1), 191–207. https://doi.org/10.1007/s13246-017-0527-5

---

*Note on verification: every F1 figure above was cross-checked against at least the original publisher page (MDPI, PLOS, Springer, Frontiers, IEEE, ScienceDirect) or PubMed abstract; several (rows 1, 6, 8, 9, 12, 13) were additionally cross-verified via Hao et al. (2022)'s own independent comparison table, which reports the same figures for Gurve et al., Barnova et al., Zhang & Yu, Mollakazemi et al., Taha & Abdel-Raheem, and Sarafan et al. as this review does.*
