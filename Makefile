.PHONY: build fetch-dev prepare-dev run run-dev validate-dev test score-dev
IMAGE ?= claushift-track2a:local
UPSTREAM_SHA ?= master
# Exact original paper prompt must be staged separately, not bundled verbatim.
PROMPT_FILE ?= prompts/paper_exact.txt
FROZEN_COMMIT ?=
# RUN_SPLIT=dev during development; switch to final with explicit confirmation after freeze.
RUN_SPLIT ?= dev
build:
	docker build -t $(IMAGE) .
fetch-dev:
	python3 -m src.cli fetch-dev --upstream-sha "$(UPSTREAM_SHA)"
prepare-dev:
	@test -n "$(SWISSGOV_ROOT)" || (echo 'Set SWISSGOV_ROOT to the cloned SwissGov-RSD directory'; exit 1)
	python3 -m src.cli prepare --split dev --upstream-root "$(SWISSGOV_ROOT)"
run: build
	@mkdir -p data/inputs out/predictions out/checkpoints
ifeq ($(RUN_SPLIT),final)
	@test -n "$(FROZEN_COMMIT)" || (echo 'FINAL denied: specify FROZEN_COMMIT=exact_git_sha'; exit 1)
	@test -f "$(PROMPT_FILE)" || (echo 'FINAL denied: stage the selected frozen prompt in $(PROMPT_FILE)'; exit 1)
	docker run --rm -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY -e UPSTREAM_SHA=$(UPSTREAM_SHA) -v "$(PWD)/data/inputs:/app/data/inputs" -v "$(PWD)/out:/app/out" $(IMAGE) sh -c 'python -m src.cli fetch-final --upstream-sha $(UPSTREAM_SHA) --confirm-final --frozen-commit $(FROZEN_COMMIT) && python -m src.cli run --split final --prompt-file $(PROMPT_FILE) --checkpoint-dir out/checkpoints/final_frozen --confirm-final --frozen-commit $(FROZEN_COMMIT) && python -m src.cli validate --split final'
else
	docker run --rm -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY -e UPSTREAM_SHA=$(UPSTREAM_SHA) -v "$(PWD)/data/inputs:/app/data/inputs" -v "$(PWD)/out:/app/out" $(IMAGE) sh -c 'python -m src.cli fetch-dev --upstream-sha $(UPSTREAM_SHA) && python -m src.cli run --split dev'
endif
run-dev: run
validate-dev:
	python3 -m src.cli validate --split dev
test:
	python3 -m unittest discover -s tests -v
score-dev:
	@test -n "$(SWISSGOV_ROOT)" || (echo 'Set SWISSGOV_ROOT'; exit 1)
	python3 -m src.cli score-dev --upstream-root "$(SWISSGOV_ROOT)"
