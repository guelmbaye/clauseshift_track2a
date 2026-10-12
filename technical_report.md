# Technical report — CLAUSESHIFT

- **Track:** 2A — UZH, SwissGov-RSD
- **Event:** Hack Apertus 2026 Online Hackathon
- **Team:** To fill in before submission
- **Demo:** Optional; not yet recorded
- **State:** Development baseline scaffold only — no measured Apertus result yet

## 1. Summary
CLAUSESHIFT aims to detect token-level cross-lingual differences on SwissGov-RSD using Apertus 1.5. The first implementation is a strict, reproducible inference and evaluator pipeline. No score is claimed before a real Apertus run.

## 2. Architecture
See `docs/DOCUMENT_01_FROZEN_V1.2.md` and `src/`. Input-only dev JSONL -> paper-style or exact paper Appendix-E prompt -> Apertus inference -> strict token projections -> encoder-compatible JSONL -> official evaluator.

## 3. Use of Apertus
- **Model:** `swiss-ai/Apertus-v1.5-8B` (pending endpoint/revision confirmation).
- **Method:** Inference with deterministic/temperature-zero decoding if supported.
- **Endpoint:** Configured by `LLM_BASE_URL`, credentials in `LLM_API_KEY`.
- **Prompt:** `PAPER_STYLE` in `src/core.py` is not the exact research prompt; for verified reproduction supply `prompts/paper_exact.txt`, containing full Appendix E.

## 4. Data
SwissGov-RSD, public benchmark. DEV only until code freeze. `data/inputs` contains unlabeled text and IDs only. Respect copyright and CC BY licence information in the research paper/dataset.

## 5. Evaluation
Primary token-level Spearman, scored separately for EN–DE, EN–FR and EN–IT, macro-averaged. Official script `scripts.evaluate_predictions_admin` with `--split dev` during development. No measured results yet.

| Setup | Mean Spearman (dev) | EN-DE | EN-FR | EN-IT | Runtime |
|---|---|---|---|---|---|
| E0: Apertus paper prompt | PENDING | PENDING | PENDING | PENDING | PENDING |
| E1: alignment-only | NOT IMPLEMENTED | - | - | - | - |
| E2: CLAUSESHIFT fusion | NOT IMPLEMENTED | - | - | - | - |

## 6. Limitations
Real Apertus inference, exact paper template ingestion and scoring are pending. Long documents may exceed CSCS's reported 60-second gateway limit; invalid/incomplete results are rejected, not silently padded.

## 7. Reproducibility
`make run` currently runs the **dev paper-style prototype** on a clean Docker build and fetches DEV-only inputs, provided public dataset access and a valid Apertus endpoint. Before official submission change it to reproduce the frozen selected method and full-order outputs. Dev/final split gates implemented in `src/cli.py`. Docker runtime unavailable in package build environment; must be tested externally.

## 8. Next steps
1. Run real E0 on dev with exact paper prompt.
2. Reproduce evaluation and report scores/costs.
3. Compare E1 vs E2, record ablations.
4. Freeze, final full-order predictions, test only after freeze.

## License
Technical report CC-BY-4.0. Code Apache-2.0.

## References
- https://aclanthology.org/2026.acl-long.1437.pdf
- https://github.com/ZurichNLP/SwissGov-RSD
- https://github.com/HackApertus/project-template/tree/main/track_2a
