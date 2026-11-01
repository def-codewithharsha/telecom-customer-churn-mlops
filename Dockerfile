FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY models/churn_model.joblib ./models/churn_model.joblib

RUN pip install --no-cache-dir --no-deps .

EXPOSE 8080

CMD ["uvicorn", "telecom_churn.api:app", "--host", "0.0.0.0", "--port", "8080", "--app-dir", "src"]
