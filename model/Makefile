# FARM-US developer shortcuts. Uses uv; run from model/.
CFG ?= configs/experiments/us_soybeans.yaml
SMOKE ?= configs/experiments/smoke_dummy.yaml
RUN = uv run

.PHONY: help sync sync-prithvi test test-int lint fmt typecheck inventory manifest \
        verify-splits leakage smoke train-smoke eval-smoke clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

sync: ## Install core + dev deps (CPU torch)
	$(RUN) --version >/dev/null; uv sync --extra dev

sync-prithvi: ## Install the real Prithvi backbone (TerraTorch, large)
	uv sync --extra prithvi --extra dev

test: ## Run unit/synthetic tests (no real backbone)
	$(RUN) pytest -m "not integration" -q

test-int: ## Run the real-Prithvi integration tests (needs [prithvi] + network)
	$(RUN) pytest -m integration -q

lint: ## ruff check
	$(RUN) ruff check src tests scripts

fmt: ## ruff format + autofix
	$(RUN) ruff format src tests scripts && $(RUN) ruff check --fix src tests scripts

typecheck: ## mypy
	$(RUN) mypy src

inventory: ## Scan on-disk data
	$(RUN) python -m farm_us.cli inventory --config configs/data/us_soybeans_hls.yaml

manifest: ## Build the chip manifest
	$(RUN) python -m farm_us.cli build-manifest --config $(CFG)

verify-splits: ## Print the resolved LOYO fold
	$(RUN) python -m farm_us.cli verify-splits --config $(CFG) test_year=2018

leakage: ## Run the leakage audit
	$(RUN) python -m farm_us.cli leakage-audit --config $(CFG) test_year=2018

smoke: ## Model smoke test (dummy backbone)
	$(RUN) python -m farm_us.cli smoke-test-model --config $(CFG)

train-smoke: ## 1-epoch synthetic training (CPU)
	$(RUN) python -m farm_us.cli train --config $(SMOKE)

eval-smoke: ## Evaluate synthetic fold
	$(RUN) python -m farm_us.cli evaluate --config $(SMOKE)

clean: ## Remove caches + outputs (keeps .gitkeep)
	rm -rf .pytest_cache .ruff_cache .mypy_cache outputs/runs outputs/manifests
