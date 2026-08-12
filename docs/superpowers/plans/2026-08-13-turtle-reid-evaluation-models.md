# Turtle Re-ID Evaluation and Model Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the prototype reaches at least 85% time-aware Rank-1 while controlling open-set false accepts, and determine whether MegaDescriptor, ArcFace fine-tuning, detector changes, or front-flipper evidence actually improve it.

**Architecture:** A manifest-driven evaluator calls the same preprocessing, embedding, retrieval, and decision services as the engineering app. Every experiment is immutable JSON plus per-query predictions. Head and front-flipper tracks are separate; fusion is an optional final experiment only if both modalities independently validate.

**Tech Stack:** Python, PyTorch/timm, Ultralytics YOLO, NumPy/SciPy/scikit-learn, OpenCV, bootstrap confidence intervals, McNemar paired tests.

**Spec:** `docs/superpowers/specs/2026-08-13-turtle-registration-reid-design.md`

## Global Constraints

- Primary success gate: time-aware known-turtle Rank-1 >= 85%; secondary: Rank-5 >= 95%.
- Open-set gate: false-accept rate <= 5% at a threshold selected without looking at the final test identities.
- Report Rank-1, Rank-5, mAP, known/unknown counts, FAR, FRR, species, side, modality, image-quality bands, and 95% bootstrap intervals.
- Split by turtle identity and chronology. Never place the same image, near-duplicate, or derived crop in both training and test.
- Results with too few repeat Olive Ridley encounters are labelled “not yet measurable,” not success.
- Compare candidates on identical query sets and use paired statistics; do not select models from headline accuracy alone.

---

## Task 1: Dataset manifests and leakage audit

**Files:** create `evaluation/manifests.py`, `evaluation/leakage.py`, `tests/test_manifests.py`, `tests/test_leakage.py`, `configs/datasets/*.json`.

- [ ] Inventory Reunion, SeaTurtleIDHeads, Zakynthos, Zindi, Amvrakikos, and any project-owned Olive Ridley photos without copying or rewriting source data.
- [ ] Normalize identity, species, date, site, side, body region, bbox/mask, and source licence/provenance.
- [ ] Hash files and perceptual thumbnails; flag exact and likely near-duplicates across splits.
- [ ] Create deterministic development/calibration/final-test partitions by identity and time.
- [ ] Commit: `test: add turtle dataset manifests and leakage audit`.

## Task 2: Evaluator parity with the app

**Files:** create `evaluation/evaluate.py`, `evaluation/metrics.py`, `evaluation/report.py`, `tests/test_metrics.py`, `tests/test_evaluator_parity.py`.

- [ ] Prove one query produces identical ranking through the evaluator and `IdentificationService` under the same registry/configuration.
- [ ] Implement CMC Rank-1/5, mAP, per-query export, open-set FAR/FRR, bootstrap intervals, and McNemar comparison.
- [ ] Emit machine-readable `summary.json`, `predictions.csv`, and an HTML/Markdown report consumed by the app.
- [ ] Commit: `feat: add reproducible open-set reid evaluation`.

## Task 3: Reproduce current baselines

- [ ] Re-run saved MegaDescriptor L/T configurations where source datasets and caches are available.
- [ ] Confirm historical figures against saved `mentah.npz`; explain any mismatch before proceeding.
- [ ] Record full-frame, ground-truth-head-crop, Zakynthos-YOLO-crop, and optional XFeat reranking separately.
- [ ] Preserve SeaTurtleIDHeads’ strong frozen baseline and do not apply XFeat where paired evidence shows harm.
- [ ] Commit only reports/configuration, never generated model caches.

## Task 4: Detector domain-shift experiment

**Files:** create `evaluation/evaluate_detector.py`, `configs/experiments/detector_cross_domain.json`.

- [ ] Measure the existing Zakynthos head detector on Zakynthos and every external dataset with compatible annotations.
- [ ] Report detection rate, IoU, crop coverage, and downstream Rank-1; call out the known Reunion transfer failure.
- [ ] If labelled data permits, train a cross-domain head detector with train identities disjoint from final evaluation identities.
- [ ] Promote a detector only when downstream Re-ID improves without unacceptable missed detections.

## Task 5: MegaDescriptor versus ArcFace metric-learning experiment

**Files:** create `training/dataset.py`, `training/train_arcface.py`, `training/losses.py`, `training/export.py`, `tests/test_training_split.py`, `configs/experiments/arcface_*.json`.

- [ ] Start with MegaDescriptor L frozen as the control.
- [ ] Add frozen embeddings plus a small normalized projection head before unfreezing the backbone.
- [ ] Fine-tune with ArcFace identity supervision on training identities only, balanced across side/species/source; validate on unseen identities/time periods.
- [ ] Compare at least: frozen MegaDescriptor; projection-only ArcFace; last-block ArcFace; and full fine-tuning only if data volume supports it.
- [ ] Track overfitting and report checkpoint selection criteria. Never reuse the failed code under `tidak_penting/`.
- [ ] Promote only statistically supported improvement on the untouched test set and acceptable open-set FAR.

## Task 6: Olive Ridley strategy

- [ ] Separate species-general pretraining data from project-specific Olive Ridley validation.
- [ ] Use legally/procedurally valid web images for detector/representation pretraining only when identity and licence metadata permit; they cannot prove field re-identification accuracy.
- [ ] Collect project-owned repeat encounters with both sides, dates, sites, and quality metadata.
- [ ] Report Olive Ridley performance separately and show confidence intervals/sample size.
- [ ] If the repeat-sighting count is inadequate, publish a collection requirement instead of an accuracy claim.

## Task 7: Front-flipper experiment

**Files:** create `evaluation/flipper_manifest.py`, `configs/experiments/flipper_megadescriptor.json`, `configs/experiments/flipper_local.json`.

- [ ] Acquire/label front-flipper crops with identity and side from datasets that permit use, including Olive Ridley where available.
- [ ] Begin with manual/ground-truth crops to isolate representation quality from detector quality.
- [ ] Benchmark frozen MegaDescriptor and a local-feature verification baseline in the `front_flipper` namespace.
- [ ] If repeat counts allow, train a flipper-specific metric head using identity-disjoint validation.
- [ ] Report flipper Rank-1/5 and open-set metrics independently; do not mix them into head scores.
- [ ] Test score-level head+flipper fusion only on encounters containing both, choosing fusion weights on calibration data. Adopt only if final-test Rank-1 improves and FAR does not worsen.

## Task 8: Calibrate user-facing evidence

**Files:** create `evaluation/calibrate.py`, `tests/test_calibration.py`, `configs/calibration/*.json`.

- [ ] Fit probability calibration from top-1 similarity, margin, quality, modality, and reference count on calibration identities only.
- [ ] Select likely/review/unknown thresholds under FAR <= 5%.
- [ ] Measure reliability diagram, Brier score, expected calibration error, FAR, and FRR.
- [ ] Export a checksummed calibration artifact tied to model, preprocessing, dataset, and split versions.
- [ ] The app displays a percentage only when this exact compatible artifact is loaded.

## Task 9: Final decision report

- [ ] Freeze all choices before opening the final test partition.
- [ ] Run the final test once per frozen candidate and publish per-query predictions.
- [ ] State one of: `passes >=85%`, `does not pass`, or `not yet measurable`; include uncertainty and open-set gate.
- [ ] Recommend the simplest model meeting both accuracy and FAR gates. If none pass, identify the highest-value next data/model experiment.
- [ ] Update the engineering app’s Evaluation tab and the canonical Notion report with sources, commands, artifacts, and limitations.

