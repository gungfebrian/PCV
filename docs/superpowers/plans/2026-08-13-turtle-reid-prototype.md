# Turtle Registration Re-ID Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a runnable engineering app that registers a turtle from one or more photos and later returns an unknown/known decision plus human-reviewable top-5 candidates.

**Architecture:** Keep the user-facing engineering console thin. A framework-independent `turtle_reid` service owns SQLite records, immutable image artifacts, crop/embedding provenance, candidate aggregation, and decisions. `aplikasi/prototipe_registrasi.py` calls that service and can later be replaced by the separate production frontend without rewriting identification logic.

**Tech Stack:** Python 3.14, NumPy, OpenCV, PyTorch/timm, SQLite, Streamlit, stdlib `unittest` and `argparse`.

**Spec:** `docs/superpowers/specs/2026-08-13-turtle-registration-reid-design.md`

## Global Constraints

- Preserve the locked historical protocol in `eksperimen/protokol.py`; new product code lives in `turtle_reid/`.
- Never compare left head, right head, and front flipper embeddings across body-region namespaces.
- Never display raw cosine similarity as an “accuracy percentage.” Before calibration, display similarity and evidence labels; after held-out calibration, display calibrated match probability with its model/calibration version.
- Unknown is a first-class result. A low-scoring query must not be forced onto a registered identity.
- Store original inputs and derived artifacts with checksums and model/preprocessing versions.
- The engineering app must expose all information needed to reproduce a result: query, crop, modality, side, model, thresholds, candidates, and timing.

---

## Task 1: Recover and freeze the verified historical baseline

**Files:** restore `CLAUDE.md`, `KONTEKS_LENGKAP.md`, `eksperimen/*.py`, `eksperimen/README.md`, `eksperimen/AUDIT.md`, and `aplikasi/{penyu_live.py,tampilan.py,uji_tampilan.py,uji_setara.py}` from commit `a70bad19a6cbdb4b6582c195b5d45bd6d43ef8cc`.

- [ ] Restore only text source files through `apply_patch`; do not overwrite datasets, weights, caches, or saved result arrays.
- [ ] Run `.venv/bin/python eksperimen/uji.py`; expected: 26 invariant tests pass.
- [ ] Run `.venv/bin/python aplikasi/uji_tampilan.py`; expected: visual-layout tests pass headlessly.
- [ ] Run the supported app/experiment equivalence test if its required embeddings are present; record an explicit skip reason otherwise.
- [ ] Commit: `chore: recover turtle reid baseline source`.

## Task 2: Define the product contracts test-first

**Files:** create `turtle_reid/__init__.py`, `turtle_reid/types.py`, `tests/test_types.py`.

- [ ] Write failing tests for `BodyRegion(head, front_flipper)`, `Side(left, right, unknown)`, `EvidenceLabel(likely_match, needs_review, unknown)`, `QueryInput`, `Candidate`, and `IdentificationResult` JSON serialization.
- [ ] Implement immutable dataclasses with validation: a head query requires a known side; a flipper query accepts left/right/unknown but is isolated from head.
- [ ] Ensure `IdentificationResult` contains `decision`, `top_candidates`, `query_artifact`, `model_version`, `calibration_version`, `thresholds`, and stage timings.
- [ ] Run `.venv/bin/python -m unittest tests.test_types -v`; expected: pass.
- [ ] Commit: `feat: define turtle reid service contracts`.

## Task 3: Build a persistent registry and immutable artifacts

**Files:** create `turtle_reid/database.py`, `turtle_reid/artifacts.py`, `tests/test_database.py`, `tests/test_artifacts.py`.

- [ ] Write failing tests for registering a turtle, rejecting duplicate public IDs, adding encounters, listing reference views, and excluding soft-disabled images.
- [ ] Implement SQLite migrations for `turtles`, `encounters`, `images`, `regions`, `embeddings`, `identification_runs`, and `review_decisions`.
- [ ] Write failing tests proving byte-identical imports deduplicate by SHA-256 while keeping encounter metadata.
- [ ] Implement `ArtifactStore.import_image`, `save_crop`, and `save_embedding`; use content-addressed paths under a configurable data root.
- [ ] Run `.venv/bin/python -m unittest tests.test_database tests.test_artifacts -v`; expected: pass.
- [ ] Commit: `feat: add turtle registry and artifact provenance`.

## Task 4: Implement region extraction and quality evidence

**Files:** create `turtle_reid/regions.py`, `turtle_reid/quality.py`, `tests/test_regions.py`, `tests/test_quality.py`.

- [ ] Define `RegionExtractor.extract(image_bgr, body_region, side) -> RegionResult` with crop, bbox, source, and warnings.
- [ ] Add a manual/full-image fallback for engineering tests and a Zakynthos YOLO head adapter that reports “no detection” instead of silently using the entire frame.
- [ ] Keep `front_flipper` manual-crop-only until a labelled detector dataset exists.
- [ ] Add deterministic blur, brightness, resolution, and region-area diagnostics; diagnostics warn but do not fabricate a match.
- [ ] Run `.venv/bin/python -m unittest tests.test_regions tests.test_quality -v`; expected: pass.
- [ ] Commit: `feat: add versioned region and image-quality pipeline`.

## Task 5: Add interchangeable embedding backends

**Files:** create `turtle_reid/embeddings.py`, `tests/test_embeddings.py`.

- [ ] Define `EmbeddingBackend.model_version` and `embed(rgb: np.ndarray) -> np.ndarray`.
- [ ] Create a deterministic fake backend and test L2 normalization, dimension checks, and NaN rejection.
- [ ] Adapt cached `BVRA/MegaDescriptor-L-384` as the first real backend without changing the historical transform.
- [ ] Add backend namespaces so future ArcFace checkpoints and flipper models cannot collide with MegaDescriptor embeddings.
- [ ] Run fake-backend tests, then a one-image offline MegaDescriptor smoke test using the local Hugging Face cache.
- [ ] Commit: `feat: add reproducible embedding backends`.

## Task 6: Retrieve identities, not individual photos

**Files:** create `turtle_reid/retrieval.py`, `tests/test_retrieval.py`.

- [ ] Write synthetic tests where multiple photos of one turtle aggregate into one candidate and cannot occupy several top-5 slots.
- [ ] Filter references by exact body-region and compatible side before cosine search.
- [ ] Implement max and top-N-mean identity aggregation as explicit configurations; default to maximum similarity for the first baseline.
- [ ] Return top-5 unique identities with best evidence image, raw similarity, margin, and rank.
- [ ] Run `.venv/bin/python -m unittest tests.test_retrieval -v`; expected: pass.
- [ ] Commit: `feat: add side-aware identity retrieval`.

## Task 7: Make the unknown decision calibratable

**Files:** create `turtle_reid/decisions.py`, `tests/test_decisions.py`.

- [ ] Write boundary tests for similarity threshold, top-1/top-2 margin, poor image quality, and empty registry.
- [ ] Implement conservative three-way decisions: `likely_match`, `needs_review`, `unknown`.
- [ ] Add a calibration artifact reader that maps features to held-out probability only when dataset/model/version checks match; otherwise omit percentage confidence.
- [ ] Confirm an unregistered synthetic identity returns `unknown` and is never automatically inserted.
- [ ] Run `.venv/bin/python -m unittest tests.test_decisions -v`; expected: pass.
- [ ] Commit: `feat: add open-set turtle decisions`.

## Task 8: Compose registration and identification services

**Files:** create `turtle_reid/service.py`, `tests/test_service.py`.

- [ ] Test `register_turtle`, `add_reference_photo`, `identify`, `confirm_candidate`, and `register_as_new` end to end with temporary storage and the fake backend.
- [ ] Registration must show extraction/quality warnings and require an explicit confirmation flag before persistence.
- [ ] Identification must log the complete run and allow later human correction without overwriting the original prediction.
- [ ] Verify adding a confirmed later encounter improves the registry without contaminating evaluation fixtures.
- [ ] Run `.venv/bin/python -m unittest tests.test_service -v`; expected: pass.
- [ ] Commit: `feat: implement turtle registration and identification service`.

## Task 9: Deliver the visible engineering app

**Files:** create `aplikasi/prototipe_registrasi.py`, `turtle_reid/rendering.py`, `turtle_reid/cli.py`, `tests/test_rendering.py`, update `README.md`.

- [ ] Build a Streamlit engineering console with pages/tabs: `Register`, `Identify`, `Registry`, `Evaluation`.
- [ ] In Register: accept turtle ID/name/species/date/site plus multiple images; let engineer select head/flipper and side, inspect crop/quality, then commit.
- [ ] In Identify: upload a later image, choose modality/side, show unknown/known/review state, calibrated probability only if available, and five unique candidate cards with raw similarity and evidence photo.
- [ ] Add controls to confirm a candidate, mark unknown, or register as new; audit every action.
- [ ] Add an evaluation panel that reads benchmark JSON rather than hard-coded metrics.
- [ ] Provide `python -m turtle_reid.cli` JSON commands for frontend integration and a contact-sheet renderer for reproducible visual QA.
- [ ] Test app import, rendering dimensions, empty registry, corrupt upload, no YOLO detection, and top-5 with fewer than five identities.
- [ ] Launch smoke test: `.venv/bin/streamlit run aplikasi/prototipe_registrasi.py --server.headless true`; expected: health endpoint responds and no startup exception.
- [ ] Commit: `feat: add turtle reid engineering prototype app`.

## Task 10: Prototype acceptance checkpoint

- [ ] Run all unit/integration tests: `.venv/bin/python -m unittest discover -s tests -v`.
- [ ] Run historical invariant and display tests again.
- [ ] Exercise one real registration and later-query flow using copied workspace photos, never modifying source datasets.
- [ ] Save the contact sheet, JSON result, app screenshot, commands, model versions, and observed limitations.
- [ ] Update the Notion implementation report with the runnable command and evidence.

