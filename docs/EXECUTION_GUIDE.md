# CLAUSESHIFT — Implementation & Execution Guide (12–16 October 2026)

> This is an **implementation starter**, not measured Apertus benchmark performance. No actual Apertus endpoint, GPU or SwissGov DEV dataset was accessible from the build environment. Tests use a **mock** OpenAI-compatible server and synthetic inputs only.

## 0. What is in this ZIP

```text
CLAUSESHIFT_Track2A/
├── README.md                          track entry and next actions
├── Makefile                           build/run Docker, validation, tests
├── Dockerfile                         Python 3.11 runtime
├── technical_report.md                aligned to official report template
├── .env.example                       environment variable names, no credentials
├── .gitignore                         no secrets, data outputs or predictions committed
├── LICENSE                            Apache 2.0 code licence
├── src/
│   ├── core.py                        prompt, strict JSON + token score conversion
│   ├── inference.py                   OpenAI-compatible Apertus adapter, timeout control
│   └── cli.py                         fetch/prepare/run/validate/score-dev/final lock
├── docs/
│   ├── DOCUMENT_01_FROZEN_V1.2.md      frozen decision brief
│   └── EXECUTION_GUIDE.md             this detailed execution plan
├── prompts/README.md                  how to supply exact paper Appendix E
├── tests/                              mock integration + invariants
├── data/inputs/                        unlabeled DEV/FINAL input files, generated
└── out/                                checkpoints + official-compatible predictions
```

## 1. Install prerequisites

### Linux / macOS / WSL2
Install Python >=3.11, Git and Docker Desktop/Engine (for `make run`), then:

```bash
unzip CLAUSESHIFT_Track2A_Implementation_v1.zip
cd CLAUSESHIFT_Track2A
python3 -m unittest discover -s tests -v
```

### Windows PowerShell
Install Python 3.11+, Git, Docker Desktop with WSL2, and unzip the package:

```powershell
cd CLAUSESHIFT_Track2A
py -3 -m unittest discover -s tests -v
```

`make` is easiest in WSL2 or Git Bash. The Python CLI works directly in PowerShell as well.

**Expected:** all local unit/integration tests pass. The integration test uses a fake HTTP model and does not validate model quality.

## 2. Obtain the upstream research code (DEV only)

```bash
git clone --depth 1 https://github.com/ZurichNLP/SwissGov-RSD.git ../SwissGov-RSD
export SWISSGOV_ROOT="$(realpath ../SwissGov-RSD)"
(cd "$SWISSGOV_ROOT" && git rev-parse HEAD)
```

The research repository **contains** test files; do not open, evaluate, search or build prompts from them during development. Only `data/evaluation/gold_labels/dev` is used by our DEV input preparation. The `fetch-dev` command independently downloads just the three DEV gold-label files and strips labels before saving prompts.

To pin an upstream commit, set `UPSTREAM_SHA` to a real Git commit SHA. If omitted, `master` is used; that is easier initially but is not immutable.

## 3. Create input-only DEV files

**Option A — from checked-out upstream (preferred for pinned reproducibility):**

```bash
python3 -m src.cli prepare --split dev --upstream-root "$SWISSGOV_ROOT"
```

**Option B — from the public DEV-only raw files (no upstream clone needed):**

```bash
python3 -m src.cli fetch-dev --upstream-sha master
```

Expected paths:

```text
data/inputs/dev_de.jsonl
data/inputs/dev_fr.jsonl
data/inputs/dev_it.jsonl
```

Each line must only contain `id`, `text_a`, `text_b`. No test data or development gold annotations are sent to Apertus.

**Verification:**

```bash
python3 - <<'PY'
import json
for lang in ('de','fr','it'):
  path=f'data/inputs/dev_{lang}.jsonl'
  with open(path,encoding='utf-8') as f:
    row=json.loads(next(f))
  assert set(row)=={'id','text_a','text_b'},row.keys()
  print(lang,row['id'],len(row['text_a'].split()),len(row['text_b'].split()))
PY
```

Only the DEV subset is inspected here. Never open the TEST subset to check correctness before freeze.

## 4. Configure real Apertus inference

Example environment (replace with your ACTUAL authorized endpoint and token):

```bash
export LLM_NAME='swiss-ai/Apertus-v1.5-8B'
export LLM_BASE_URL='https://api.swissai.svc.cscs.ch/v1'
export LLM_API_KEY='YOUR_OWN_AUTHORIZED_KEY'
```

**Do not commit a `.env` containing credentials.** Another endpoint is acceptable if it serves the required Apertus model through OpenAI-compatible `/chat/completions`. Model identity must be checked in provider logs / model metadata, not just assumed from the client model string.

Ping the endpoint separately with an authorized request before expensive long prompts. Do not publish HTTP authorization headers or complete API responses containing sensitive metadata.

The tool uses an HTTP timeout of 52 seconds to avoid reported CSCS 60s gateway timeouts. HTTP 504 fails immediately; streaming alone is not a workaround. If a document is too long, use a separately documented chunking method (NOT an unchanged reproduction of Appendix E).

## 5. Stage the EXACT research-paper prompting baseline

The UZH researchers ask for a reproduced baseline from the SwissGov paper, using Apertus.

Appendix E of https://aclanthology.org/2026.acl-long.1437.pdf supplies the three-shot baseline prompt.

1. Obtain Appendix E directly from the paper.
2. Copy its complete prompt, examples, scale and formatting into `prompts/paper_exact.txt` (UTF-8).
3. Keep `{{ sentence1 }}` and `{{ sentence2 }}` as literal replacement placeholders.
4. Record the SHA-256 of this prompt in the technical report.
5. Use the exact template with `--prompt-file prompts/paper_exact.txt`.

The built-in `PAPER_STYLE` prompt is **independently authored** with three examples for integration testing; it is **NOT** a verbatim reproduction and its results must be labelled *paper-style prompt baseline*, not *exact paper prompt baseline*. This distinction is important for scientific honesty.

The runner converts model similarity scores to divergence with `d = 1 - s/5` and retains `-1` for punctuation. It rejects missing, extra or modified tokens rather than silently padding with zero.

## 6. Smoke-test REAL inference before complete benchmark

Use separate checkpoint directories for any change of prompt, model or output length.

```bash
python3 -m src.cli run \
  --split dev \
  --prompt-file prompts/paper_exact.txt \
  --max-examples 1 \
  --max-tokens 4096 \
  --checkpoint-dir out/checkpoints/exact_v1
```

`--max-examples 1` means one **new** input per language; no complete prediction file is exported until every DEV row has been processed. Three HTTP calls (one per language) are expected on a clean checkpoint.

Inspect errors, latency and model outputs. The runner does not fabricate scores when the model emits malformed JSON, fails or times out.

If the 60s gateway limit is reached, decide whether to host Apertus with your own container or switch to an explicitly labelled chunked alternative. Do not keep retrying requests that inevitably exceed gateway duration.

## 7. Full baseline on DEV

```bash
python3 -m src.cli run \
  --split dev \
  --prompt-file prompts/paper_exact.txt \
  --max-tokens 4096 \
  --checkpoint-dir out/checkpoints/exact_v1
```

**Outputs on completion:**

```text
out/predictions/CLAUSESHIFT_admin_de.jsonl.jsonl
out/predictions/CLAUSESHIFT_admin_fr.jsonl.jsonl
out/predictions/CLAUSESHIFT_admin_it.jsonl.jsonl
```

The double extension is intentional: the original `evaluate_predictions_admin.py` appends `.jsonl.jsonl` to non-LLM output prefixes. Our predictions use the official evaluator's `EncoderPrediction` JSON shape, despite being produced by an LLM. This avoids the separate LLM response-parser branch. Check actual evaluator behavior if upstream changes.

**IMPORTANT:** an interrupted run leaves a checkpoint. A second execution continues from completed items. The final files are only generated when 100% of expected records have been processed and validated.

## 8. Strict input/output validation

```bash
python3 -m src.cli validate --split dev
```

The checker rejects bad ID/order, missing outputs, mismatched text, wrong number of token scores, out-of-range scores, NaNs and duplicates. The reference evaluator can pad/truncate in some cases; this validator intentionally refuses such cases.

## 9. Run the ORIGINAL official scorer on DEV

The reference script depends on its original packages. In a separate Python venv (Python 3.11 preferred):

```bash
python3.11 -m venv .venv-rsd
source .venv-rsd/bin/activate
python -m pip install -r "$SWISSGOV_ROOT/requirements.txt"
python -m pip install -r "$SWISSGOV_ROOT/rsd/requirements.txt"
python -m src.cli score-dev --upstream-root "$SWISSGOV_ROOT"
```

The original command run by our wrapper is equivalent to:

```bash
(cd "$SWISSGOV_ROOT" && python -m scripts.evaluate_predictions_admin \
  /absolute/path/to/CLAUSESHIFT_Track2A/out/predictions/CLAUSESHIFT_admin_ --split dev)
```

The tool deliberately exposes **only** `score-dev`. The `--split test` call must be reserved for the frozen final phase. Record EN-DE / EN-FR / EN-IT Spearman, macro average and Kendall tau-b in `technical_report.md`, with model revision, prompt hash and inference cost. Do not invent numeric results.

## 10. Docker `make run` from a clean project checkout

```bash
export LLM_NAME='swiss-ai/Apertus-v1.5-8B'
export LLM_BASE_URL='YOUR_AUTHORIZED_APERTUS_ENDPOINT/v1'
export LLM_API_KEY='YOUR_KEY'
make run
```

Current `make run` builds a Python 3.11 image, fetches **DEV only** directly from SwissGov-RSD, and performs DEV inference. It **intentionally uses the self-contained paper-style scaffold**, not an unpublished exact Appendix-E prompt. For exact reproduction, run the explicit CLI above or modify the entrypoint to mount the exact template once it is staged. Docker itself was unavailable in the authoring environment; local Python integration tests passed, Docker runtime requires user-side verification.

The Makefile now has a gated `RUN_SPLIT=final` path for full-order E0 predictions after freeze. If the selected winning method is not E0, `make run` MUST be adapted accordingly and tested from a clean checkout. The current development-only `make run` is not sufficient to claim submission readiness.

## 11. Candidate improvement after E0 is measured

Implement E1–E4 in this order:

1. Alignment-only strong comparator (prefer authentic DiffAlign implementation if feasible).
2. Candidate ranking of potential additions/omissions, with alignment offsets.
3. Apertus adjudication on selected cross-lingual contexts.
4. Exact projection of adjusted difference values back to original tokens.
5. Rank-aware fusion tuned only on DEV.
6. Ablation with Apertus disabled; same evaluation and runtime measures.

Add each as a new method flag/module, with separate checkpoints, explicit config hash and full evaluation. Do not contaminate E0 baseline with method changes.

## 12. Final freeze and TEST-only procedure — NEVER during development

After method selection on DEV and document/code freeze:

```bash
git status --porcelain
GIT_SHA=$(git rev-parse HEAD)
printf '%s\n' "$GIT_SHA" > docs/frozen_commit.txt
```

Then, and **only then**, materialize the full-order official inputs. The preparation implementation uses `full/gold_admin_*.jsonl` after freeze but **extracts only id/text fields**, never uses test labels for inference or tuning. If an official dedicated input-only full dataset is available, prefer it and validate identity/order. Because these reference files also contain labels, treat this extraction step as *final evaluation infrastructure only*, do not run it during model development.

```bash
python -m src.cli prepare --split final --upstream-root "$SWISSGOV_ROOT" \
  --confirm-final --frozen-commit "$GIT_SHA"

python -m src.cli run --split final \
  --prompt-file prompts/paper_exact.txt \
  --checkpoint-dir out/checkpoints/final_frozen \
  --confirm-final --frozen-commit "$GIT_SHA"

python -m src.cli validate --split final
```

Alternatively, after freeze, `make run RUN_SPLIT=final FROZEN_COMMIT="$GIT_SHA" PROMPT_FILE=prompts/paper_exact.txt` runs an automated full-order FINAL pass with Docker, if the real endpoint and original prompt are configured. **Never run this during development**. The build's ability to run the final branch has not been tested with Docker here.

**Manual frozen-only official test evaluation**, once the full-order files are complete:

```bash
(cd "$SWISSGOV_ROOT" && python -m scripts.evaluate_predictions_admin \
  /absolute/path/to/CLAUSESHIFT_Track2A/out/predictions/CLAUSESHIFT_admin_ --split test)
```

Do not use the scored test examples, metrics or errors to tune the method. If the run fails, distinguish **infrastructure repair** from **model improvement**; record any repairs and preserve a credible audit trail. The user-provided Q&A says judges will score test while requiring full-set order.

## 13. Submission checklist

- [ ] Original HackApertus `track_2a/` layout preserved and reconciled with latest upstream commit.
- [ ] **Exact** paper Appendix-E prompt staged and cited (or method difference explicitly acknowledged).
- [ ] Real Apertus 8B/70B endpoint used and model identity verified.
- [ ] E0 baseline scored on dev with the official script.
- [ ] E1–E4 improvements, if claimed, actually implemented and scored.
- [ ] No test used before final code freeze.
- [ ] Three full-order prediction files validated without padding/truncation.
- [ ] `make run` adapted to selected **final** method, verified with Docker clean checkout.
- [ ] No API keys or proprietary logs committed.
- [ ] Data redistribution and third-party copyrights respected.
- [ ] Technical report includes valid numbers, runtime, hardware and limits.
- [ ] Code Apache-2.0 and docs CC-BY-4.0; prompt/source attributions documented.
- [ ] Submitted at https://hackapertus.ch/online-hack/submissions (not on Devpost).

## 14. Blockers / evidence status at package creation

| Item | Status |
|---|---|
| Static code + schema design | Implemented |
| JSONL and token-safety automated tests | Passed on synthetic fixtures |
| Mock OpenAI-compatible inference integration | Passed for DE/FR/IT |
| GitHub project-template clone | Not possible in authoring container due outbound network/DNS restriction |
| Real SwissGov DEV download/run | Not performed in authoring container |
| Exact Appendix E template loaded | Manual user-side step pending |
| Real Apertus inference | Pending authorized endpoint |
| Official dev Spearman | Pending real predictions + upstream environment |
| Docker execution | Not tested; Docker binary unavailable in authoring container |
| Test scoring | Intentionally not performed |

**Integrity statement:** neither fictitious benchmark metrics nor test-data-driven optimization have been produced in this package.
