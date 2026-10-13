.PHONY: run build run-dev freeze preflight test validate-dev score-dev prepare-dev
IMAGE ?= clauseshift-track2a:final
UPSTREAM_SHA ?=
FROZEN_COMMIT ?=
PROMPT_FILE ?= prompts/paper_exact.txt
SWISSGOV_ROOT ?=

build:
	docker build -t "$(IMAGE)" .

# Official clean-checkout entry point: frozen E0-PJQ, full-order outputs, Docker.
# No gold-based test evaluation or tuning. Requires a committed freeze manifest.
run: build
	@mkdir -p data/inputs out/predictions out/checkpoints
	docker run --rm \
	  -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY -e LLM_USER_AGENT \
	  -v "$(PWD)/data/inputs:/app/data/inputs" \
	  -v "$(PWD)/out:/app/out" \
	  "$(IMAGE)" python -m src.submission run

# All these commands are DEV-only and have no final/test access.
run-dev: build
	@mkdir -p data/inputs out
	docker run --rm \
	  -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY -e LLM_USER_AGENT \
	  -v "$(PWD)/data/inputs:/app/data/inputs" \
	  -v "$(PWD)/out:/app/out" \
	  "$(IMAGE)" sh -c 'python -m src.cli fetch-dev --upstream-sha master && python -m src.cli run --split dev --prompt-file prompts/paper_exact.txt --punctuation-repair --syntax-repair --boundary-quote-repair --max-tokens 4096 --checkpoint-dir out/checkpoints/dev_e0pjq'

freeze:
	@test -n "$(UPSTREAM_SHA)" && test -n "$(FROZEN_COMMIT)" || (echo 'Pass exact UPSTREAM_SHA and FROZEN_COMMIT'; exit 1)
	python3 -m src.submission freeze --upstream-sha "$(UPSTREAM_SHA)" --frozen-commit "$(FROZEN_COMMIT)"
preflight:
	python3 -m src.submission preflight
prepare-dev:
	@test -n "$(SWISSGOV_ROOT)" || (echo 'Set SWISSGOV_ROOT'; exit 1)
	python3 -m src.cli prepare --split dev --upstream-root "$(SWISSGOV_ROOT)"
validate-dev:
	python3 -m src.cli validate --split dev
test:
	python3 -m unittest discover -s tests -v
score-dev:
	@test -n "$(SWISSGOV_ROOT)" || (echo 'Set SWISSGOV_ROOT'; exit 1)
	python3 -m src.cli score-dev --upstream-root "$(SWISSGOV_ROOT)"
