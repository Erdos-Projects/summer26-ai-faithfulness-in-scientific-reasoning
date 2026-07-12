# Public Review Iteration Log

This file records each independent public-review round for the SciVer router
release package, the substantive reviewer comments, and whether each comment was
accepted or rejected. Rejected comments include the reason.

## Round 1 - Independent Public Review

Date: 2026-07-09

Review surface:

- `dist/public_router_release/`
- `PUBLIC_RELEASE_MANIFEST.md`
- `docs/sciver_router_presentation.html`
- `outputs/sciver_router_models_5min.pptx`
- `outputs/sciver_router_models_5min_montage.png`
- `docs/public_data_statement.md`
- `docs/reproduce_router_experiments.md`

Reviewers:

- Public Reproducibility Reviewer
- Scientific/Methods Reviewer
- Presentation/Deck Reviewer

### Decisions

| Reviewer comment/advice | Decision | Implementation or rejection reason |
| --- | --- | --- |
| Add a release-local dependency file because the public bundle referenced `requirements.txt` without including it. | Accepted | `requirements.txt` is copied into `dist/public_router_release/`. |
| Public reproduction should start from `router_dataset_public.csv`, not from `prepare_public_router_release.py`. | Accepted | `docs/reproduce_router_experiments.md`, `docs/sciver_router_presentation.html`, and the public README now use `verify_public_router_release.py`, `run_public_router_experiments.py`, and `build_sciver_router_presentation_report.py` as the public path. |
| Replace the bundled README with a public-router-only README. | Accepted | Added `docs/public_router_release_README.md`; the release packager copies it to `dist/public_router_release/README.md`. |
| Add a public verification step for row counts, forbidden columns, comparison rows, headline metrics, and run inventory. | Accepted | Added `scripts/verify_public_router_release.py`; it verifies repo-local and release-bundle artifacts. |
| Add checksums or expected metric snippets for public CSV/comparison outputs. | Accepted | `scripts/prepare_public_router_release.py` writes `public_router_checksums.json`; the verifier checks checksums when present. |
| Clarify `validation-origin` and `test-origin` as source labels, not router evaluation splits. | Accepted | Updated the HTML report, reproduction docs, public data statement, and slide 3. |
| Add methods table with row counts, paper groups, seed, grouped CV, and bootstrap settings. | Accepted | Added methods table to `docs/sciver_router_presentation.html`. |
| Clarify random versus deterministic baselines. | Accepted | HTML report now separates seeded random from deterministic train-global-best and always-model baselines. |
| Add bootstrap intervals to the deck result slide and avoid overclaiming. | Accepted | Slide 7 now says “best point estimate” and shows top-1 and regret bootstrap intervals. |
| Put the internal grouped-holdout caveat earlier in the deck and report. | Accepted | Slide 1 includes the caveat; the HTML report adds a “Read This First” box. |
| Define TDA, top-1 hit rate, and mean regret in plainer deck language. | Accepted | Slides 4 and 5 were rewritten with plain-language definitions. |
| Add a visible deck pointer to the HTML evidence appendix. | Accepted | Slide 8 points to `docs/sciver_router_presentation.html`. |
| Add speaker notes for the five-minute deck. | Accepted | `scripts/build_sciver_router_deck.mjs` writes speaker notes into the PPTX. |
| Add feature dictionary and explain why public sidecars can have different row counts than the modeling table. | Accepted | Added to HTML report and `docs/public_data_statement.md`. |
| Add non-sensitive provenance for observed VLM accuracy targets. | Rejected for now | The current public package does not include a clean, complete, non-sensitive provenance table for all local VLM runs. Adding one would require a separate provenance audit to avoid accidentally exposing private local runtime details. The caveat that raw local inference is not public remains explicit. |
| Consider external figure assets instead of one large inline HTML file. | Rejected | Portability was the explicit presentation goal; the single-file, click-to-zoom HTML remains preferable for sharing. The report generator remains the maintainable source of truth. |
| Remove or mark private-capable legacy code paths in public scripts. | Partially accepted | Public release scripts are sanitized for public defaults and no longer contain internal subset names or excluded-model defaults. Some generic safety strings such as forbidden column names remain in the verifier by design. |

### Verification After Round 1

- `scripts/verify_public_router_release.py --root .`: passed.
- `scripts/verify_public_router_release.py --root dist/public_router_release`: passed.
- Unit tests: 81 passed.
- PPTX rendered successfully and overflow test passed.
- Public bundle scan found no internal subset names, excluded-model references, raw JSONL, DB, or SQLite files.

## Round 2 - Independent Public Review

Date: 2026-07-09

Review surface:

- `dist/public_router_release/`
- `PUBLIC_RELEASE_MANIFEST.md`
- `docs/sciver_router_presentation.html`
- `outputs/sciver_router_models_5min.pptx`
- `outputs/sciver_router_models_5min_montage.png`
- `docs/public_data_statement.md`
- `docs/reproduce_router_experiments.md`

Reviewers:

- Public Reproducibility Reviewer
- Scientific/Methods Reviewer
- Presentation/Deck Reviewer

### Decisions

| Reviewer comment/advice | Decision | Implementation or rejection reason |
| --- | --- | --- |
| Clarify why the deck feature-count chart showed fewer TDA/NLP columns than the HTML feature dictionary. | Accepted | The report and deck now label those counts as router-used feature columns, excluding status metadata. |
| Add or explicitly caveat uncertainty for baseline comparisons. | Accepted | The HTML now states that baseline comparisons are point estimates and baseline deltas are not separately uncertainty-tested. |
| Reword “meaningfully different from always choosing one VLM” because performance separation was not uncertainty-tested. | Accepted | The conclusion now says the router is behaviorally different and frames selected-model diversity as a diversity observation. |
| Keep the chart-TDA conclusion tied to point estimates rather than definitive superiority. | Accepted | The conclusion now says XGBoost + chart TDA yielded the best grouped-holdout point estimate in this run. |
| Clarify that `--skip-existing` is a quick verification path, not the true rerun path. | Accepted | The public README, reproduction doc, and report command block now tell users to omit `--skip-existing` for a true rerun in a fresh release copy. |
| Trim public dependencies so reviewers do not install private/non-router packages. | Accepted | Added `requirements-public.txt`; the release packager copies it into the bundle as `requirements.txt`. |
| Update `compare_router_models.py` defaults so direct public use points to public router outputs. | Accepted | Defaults now point to `Data/derived/model_router/public_router/...` paths. |
| Expand release verifier private-file scan to include root-level convenience copies in the release bundle. | Accepted | The verifier scans the full root when the target looks like `dist/public_router_release`; repo-root verification still scans the public surface only. |
| Add the public review iteration log to the public manifest/release if it is intended for reviewers. | Accepted | `docs/public_review_iterations.md` is now listed in the manifest and copied into the release bundle. |
| Enlarge slide 1 caveat and slide 7 labels if possible. | Accepted | The deck generator slightly enlarges the slide 1 caveat text and simplifies slide 7 data labels to one decimal place. |

### Verification After Round 2

- `scripts/verify_public_router_release.py --root .`: passed.
- `scripts/verify_public_router_release.py --root dist/public_router_release`: passed.
- Unit tests: 81 passed.
- PPTX rendered successfully and overflow test passed.
- Public bundle scan found no internal subset names, excluded-model references, raw JSONL, DB, or SQLite files.

## Round 3 - Independent Public Review

Date: 2026-07-09

Review surface:

- `dist/public_router_release/`
- `PUBLIC_RELEASE_MANIFEST.md`
- `docs/sciver_router_presentation.html`
- `outputs/sciver_router_models_5min.pptx`
- `outputs/sciver_router_models_5min_montage.png`
- `docs/public_data_statement.md`
- `docs/reproduce_router_experiments.md`
- `docs/public_review_iterations.md`

Reviewers:

- Public Reproducibility Reviewer
- Scientific/Methods Reviewer
- Presentation/Deck Reviewer

### Decisions

| Reviewer comment/advice | Decision | Implementation or rejection reason |
| --- | --- | --- |
| Clarify that the headline “best” run is selected post-hoc across the public comparison grid. | Accepted | The HTML now states that the headline run is selected after comparing the public grid and that intervals are conditional on the selected run. |
| Clarify that the twelve cognitive dimensions are human-coded annotations, not automatically generated deployment-time features. | Accepted | The HTML and public data statement now describe them as precomputed human-coded annotations and state that deployment would need annotations or an automatic proxy. |
| Add a caveat that five-run observed VLM accuracies make per-item best-model labels and regret noisy. | Accepted | The HTML definitions now state that five-repeat targets are noisy and that bootstrap intervals do not include target-estimation uncertainty. |
| Document claim-NLP feature fitting provenance. | Accepted | The HTML and public data statement now label TF-IDF/SVD and token-vector NLP features as transductive exploratory features fit once before the router split. |
| Clarify dependency reproducibility because public requirements used lower bounds while the verifier checks exact headline metrics. | Accepted | Added `requirements-lock.txt`, and docs point exact-version reproduction to it. |
| Remove unused `gudhi` from the public dependency file because public TDA features are precomputed. | Accepted | `requirements-public.txt` no longer includes `gudhi`. Private feature-extraction scripts still depend on it outside the public rerun path. |
| Make serialized model-artifact safety warnings consistently mention both `.joblib` and `.pt`. | Accepted | Updated the public README, reproduction docs, data statement, and HTML generator. |
| Extend verifier to compare root-level convenience copies against nested canonical public CSV/JSON files. | Accepted | `verify_public_router_release.py` now checks root-copy SHA-256 equality for the public dataset, feature sidecars, and audit file. |
| Add a top navigation/TOC to the HTML and enlarge small deck charts. | Rejected for now | The presentation reviewer marked these as optional polish and explicitly reported no substantial/direct deck fixes. The current deck passed render/overflow QA and the HTML sections are already linear for a five-minute evidence appendix. |
| Add paired grouped-bootstrap deltas against baselines. | Rejected for now | Useful future analysis, but not required for public readiness because the report now explicitly treats baseline comparisons as point estimates and does not claim statistically separated baseline deltas. |

### Verification After Round 3

- `scripts/verify_public_router_release.py --root .`: passed.
- `scripts/verify_public_router_release.py --root dist/public_router_release`: passed.
- Unit tests: 81 passed.
- PPTX rendered successfully and overflow test passed.
- Public bundle scan found no internal subset names, excluded-model references, raw JSONL, DB, or SQLite files.
- Public dependency file is router-only; exact tested versions are recorded in `requirements-lock.txt`.
