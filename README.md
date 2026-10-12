# CLAUSESHIFT — Hack Apertus Track 2A / University of Zurich

*Evidence-Grounded Cross-Lingual Semantic Difference Recognition with Apertus 1.5*

> **Status: implemented benchmark/baseline scaffold, not a completed or scored submission.**

Commencer en français par [`docs/GUIDE_EXECUTION_FR.md`](docs/GUIDE_EXECUTION_FR.md). For detailed English instructions see [`docs/EXECUTION_GUIDE.md`](docs/EXECUTION_GUIDE.md) for all steps (dev inputs, exact prompting, Apertus API, prediction format, validation, official evaluation, final freeze and test isolation). The frozen strategy is [`docs/DOCUMENT_01_FROZEN_V1.2.md`](docs/DOCUMENT_01_FROZEN_V1.2.md).

## Five first commands

```bash
python3 -m unittest discover -s tests -v
python3 -m src.cli fetch-dev --upstream-sha master
export LLM_NAME='swiss-ai/Apertus-v1.5-8B'
export LLM_BASE_URL='YOUR_AUTHORIZED_APERTUS_ENDPOINT/v1'
export LLM_API_KEY='YOUR_TOKEN'
python3 -m src.cli run --split dev --max-examples 1
```

Before claiming exact baseline reproduction, supply verbatim Appendix-E prompt in `prompts/paper_exact.txt` and use `--prompt-file`. The built-in prompt is only a task-aligned three-shot approximation.

## Key result to report

`macro_spearman = (rho_en_de + rho_en_fr + rho_en_it) / 3`

**No real Apertus results have been generated in this starter.**

## Official rules and research

- [HackApertus project template](https://github.com/HackApertus/project-template)
- [Track 2A template README](https://github.com/HackApertus/project-template/tree/main/track_2a)
- [UZH SwissGov-RSD task](https://hackapertus.notion.site/track-2a-uzh)
- [SwissGov-RSD official research code](https://github.com/ZurichNLP/SwissGov-RSD)
- [Paper and Appendix E](https://aclanthology.org/2026.acl-long.1437.pdf)

Code is Apache-2.0. Original scholarly/government documents and their gold annotations retain their own licences and attribution conditions. Documents prepared for project submission should be licensed CC-BY-4.0 where applicable.
