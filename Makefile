.PHONY: venv install sota test demo lint format ci

venv:
	python -m venv .venv && .venv/Scripts/pip install -U pip

install:
	pip install -r requirements.txt

sota:
	pip install "netcal>=1.4" || echo "netcal 不可用, 已降级纯 numpy 校准"

test:
	pytest -q -W ignore::UserWarning

demo:
	python -m oodforge.examples.run_demo

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check --fix .

ci: lint test
