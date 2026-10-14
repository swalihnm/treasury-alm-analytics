PY ?= python
.PHONY: install data curves betas train backtest nii irrbb montecarlo report test lint typecheck serve all
install:
	$(PY) -m pip install -e ".[dev]"
data:
	$(PY) -m treasury_alm.cli data
curves:
	$(PY) -m treasury_alm.cli curves
betas:
	$(PY) -m treasury_alm.cli betas
train backtest:
	$(PY) -m treasury_alm.cli backtest
nii:
	$(PY) -m treasury_alm.cli nii
irrbb:
	$(PY) -m treasury_alm.cli irrbb
montecarlo:
	$(PY) -m treasury_alm.cli montecarlo
report:
	$(PY) -m treasury_alm.cli report
test:
	$(PY) -m pytest --cov=treasury_alm --cov-report=term-missing
lint:
	ruff check src tests && ruff format --check src tests
typecheck:
	mypy src/treasury_alm
serve:
	$(PY) -m treasury_alm.cli serve
all: data curves betas backtest nii irrbb montecarlo report
