.PHONY: run build run-dev freeze preflight test validate-dev score-dev prepare-dev
IMAGE ?= clauseshift-track2a:final
UPSTREAM_SHA ?=
METHOD ?= E0-PJQ
EMB_SHA ?=
BATCH_CAP ?= 96
UNITS_PER_BATCH ?= 3
WORKERS ?= 1
LAUNCH_INTERVAL ?= 0
FROZEN_COMMIT ?=
PROMPT_FILE ?= prompts/paper_exact.txt
SWISSGOV_ROOT ?=

build:
	docker build -t "$(IMAGE)" .

# Official clean-checkout entry point: frozen E0-PJQ or E2-BATCH, full-order outputs, Docker.
# No gold-based test evaluation or tuning. Requires a committed freeze manifest.
run: build
	@mkdir -p data/inputs out/predictions out/checkpoints
	docker run --rm \
	  -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY -e LLM_USER_AGENT -e HF_TOKEN \
	  -v "$(PWD)/data/inputs:/app/data/inputs" \
	  -v "$(PWD)/out:/app/out" \
	  "$(IMAGE)" python -m src.submission run

# All these commands are DEV-only and have no final/test access.
run-dev: build
	@mkdir -p data/inputs out
	docker run --rm \
	  -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY -e LLM_USER_AGENT -e HF_TOKEN \
	  -v "$(PWD)/data/inputs:/app/data/inputs" \
	  -v "$(PWD)/out:/app/out" \
	  "$(IMAGE)" sh -c 'python -m src.cli fetch-dev --upstream-sha master && python -m src.cli run --split dev --prompt-file prompts/paper_exact.txt --punctuation-repair --syntax-repair --boundary-quote-repair --max-tokens 4096 --checkpoint-dir out/checkpoints/dev_e0pjq'

freeze:
	@test -n "$(UPSTREAM_SHA)" && test -n "$(FROZEN_COMMIT)" || (echo 'Pass exact UPSTREAM_SHA and FROZEN_COMMIT'; exit 1)
	python3 -m src.submission freeze --upstream-sha "$(UPSTREAM_SHA)" --frozen-commit "$(FROZEN_COMMIT)" \
	  --method "$(METHOD)" --embedding-revision "$(EMB_SHA)" \
	  --batch-cap "$(BATCH_CAP)" --units-per-batch "$(UNITS_PER_BATCH)" \
	  --workers "$(WORKERS)" --launch-interval "$(LAUNCH_INTERVAL)" \
	  --max-tokens $(if $(filter E2-BATCH,$(METHOD)),1536,4096)
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
