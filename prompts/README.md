# Prompt provenance

Default `PAPER_STYLE` prompt in `src/core.py` is an independently authored, task-aligned three-shot scaffold.
It is NOT the verbatim few-shot prompt in Appendix E of the SwissGov-RSD ACL paper.

To reproduce the literal paper baseline, obtain Appendix E from:
https://aclanthology.org/2026.acl-long.1437.pdf
and save a faithful machine-readable prompt in a local UTF-8 file, preserving both literal placeholders:
`{{ sentence1 }}` and `{{ sentence2 }}`.

Run `python -m src.cli run --prompt-file prompts/paper_exact.txt` with a NEW checkpoint directory.
Do not claim exact reproduction until the prompt, generation parameters and token mapping are verified.
