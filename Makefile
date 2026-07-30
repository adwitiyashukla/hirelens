.DEFAULT_GOAL := help
PY := python

.PHONY: help
help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install:  ## install the package and dev tooling
	$(PY) -m pip install -e ".[dev]"

.PHONY: install-all
install-all:  ## install everything including retrieval, api and eval extras
	$(PY) -m pip install -e ".[dev,retrieval,api,evals]"

.PHONY: lint
lint:  ## ruff check + format check
	ruff check src tests
	ruff format --check src tests

.PHONY: fmt
fmt:  ## autoformat
	ruff format src tests
	ruff check --fix src tests

.PHONY: types
types:  ## mypy strict
	mypy src

.PHONY: test
test:  ## unit tests (no network)
	pytest

.PHONY: test-live
test-live:  ## tests that hit a real provider
	HIRELENS_RUN_LIVE_TESTS=1 pytest -m live

.PHONY: check
check: lint types test  ## everything CI runs

.PHONY: golden
golden:  ## render the golden set so you can read what you are labelling
	$(PY) -m hirelens.evals.cli generate

.PHONY: label
label:  ## assign human screening tiers to the golden set (only you can do this)
	$(PY) -m hirelens.evals.cli label

.PHONY: eval
eval:  ## run the evaluation harness against the golden set
	$(PY) -m hirelens.evals.cli run

.PHONY: eval-smoke
eval-smoke:  ## exercise the harness with no API key (validates plumbing, not quality)
	$(PY) scripts/smoke_eval.py

.PHONY: eval-gate
eval-gate:  ## run the harness and fail on regression (used by CI)
	$(PY) -m hirelens.evals.cli run --gate

.PHONY: audit-plan
audit-plan:  ## show the audit experiment matrix and cost estimate (spends nothing)
	$(PY) -m hirelens.audit.cli plan

.PHONY: audit
audit:  ## run the counterfactual fairness audit
	$(PY) -m hirelens.audit.cli run

.PHONY: audit-smoke
audit-smoke:  ## prove the audit catches injected bias, with no API key
	$(PY) scripts/smoke_audit.py

.PHONY: audit-gate
audit-gate:  ## run the audit and fail the build on excess drift
	$(PY) -m hirelens.audit.cli run --gate

.PHONY: api
api:  ## run the API locally on SQLite, no Docker needed
	uvicorn hirelens.api.app:app --reload --port 8000

.PHONY: web-install
web-install:  ## install the dashboard's dependencies
	cd web && npm install

.PHONY: web
web:  ## run the dashboard in dev mode on :5173, proxying the API on :8000
	cd web && npm run dev

.PHONY: web-test
web-test:  ## typecheck and unit-test the dashboard
	cd web && npm run typecheck && npm test

.PHONY: web-build
web-build:  ## build the dashboard so the API serves it at http://localhost:8000
	cd web && npm run build

.PHONY: demo-set
demo-set:  ## render 4 synthetic candidates into samples/ for the dashboard demo (free)
	$(PY) scripts/make_demo_set.py

.PHONY: demo
demo: web-build  ## build the dashboard, then serve everything from one process
	uvicorn hirelens.api.app:app --port 8000

.PHONY: up
up:  ## run the API and Postgres in Docker
	docker compose up --build

.PHONY: clean
clean:  ## remove caches and build artefacts
	rm -rf .hirelens_cache .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
