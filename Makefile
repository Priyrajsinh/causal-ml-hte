.PHONY: install lint test train evaluate serve gradio streamlit audit ci docker-build

PY := venv/Scripts/python
PIP := venv/Scripts/pip

install:
	$(PIP) install -U pip wheel
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	$(PY) -m pre_commit install

lint:
	$(PY) -m black src/ tests/
	$(PY) -m isort src/ tests/ --profile black
	$(PY) -m flake8 src/ tests/
	$(PY) -m mypy src/

test:
	$(PY) -m pytest tests/ -v --tb=short --cov=src --cov-fail-under=70

train:
	$(PY) -m src.training.train --config config/config.yaml

evaluate:
	$(PY) -m src.evaluation.evaluate --config config/config.yaml

serve:
	$(PY) -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

gradio:
	$(PY) -m src.api.gradio_demo

streamlit:
	$(PY) -m streamlit run app.py

# Known unfixable CVEs — see MANUAL_TASKS.md
# PYSEC-2024-{274,271,277}: gradio / flask-cors / joblib — no fix versions
# PYSEC-2026-161: starlette<1; fix is 1.0.1 but breaks mlflow-skinny + prometheus-fastapi-instrumentator
PIP_AUDIT_IGNORE := --ignore-vuln PYSEC-2024-274 --ignore-vuln PYSEC-2024-271 --ignore-vuln PYSEC-2024-277 --ignore-vuln PYSEC-2026-161

audit:
	$(PY) -m pip_audit -r requirements.txt $(PIP_AUDIT_IGNORE)
	$(PY) -m detect_secrets scan --baseline .secrets.baseline
	$(PY) -m bandit -r src/ -ll -ii

ci:
	$(PY) -m black --check src/ tests/
	$(PY) -m isort --check-only src/ tests/ --profile black
	$(PY) -m flake8 src/ tests/
	$(PY) -m mypy src/
	$(PY) -m bandit -r src/ -ll -ii
	$(PY) -m radon cc src/ -nc
	$(PY) -m interrogate src/ --fail-under=80
	$(PY) -m pip_audit -r requirements.txt $(PIP_AUDIT_IGNORE)
	$(PY) -m detect_secrets scan --baseline .secrets.baseline
	$(PY) -m pytest tests/ -v --tb=short --cov=src --cov-fail-under=70
	@echo "All CI gates green. Safe to git push."

docker-build:
	docker build -t causal-ml-hte:latest .
