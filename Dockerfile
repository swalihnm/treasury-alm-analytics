FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY dashboard ./dashboard
RUN pip install --no-cache-dir -e .
# data/processed and results/ are mounted or rebuilt with `alm data` etc. (FRED_API_KEY required for `alm data`)
CMD ["uvicorn", "treasury_alm.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
