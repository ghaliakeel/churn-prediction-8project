FROM python:3.11-slim

WORKDIR /app

# install uv for faster package management
RUN pip install uv

# copy project files
COPY pyproject.toml .
COPY src/ src/
COPY models/ models/

# install dependencies
RUN uv pip install --system -e .

EXPOSE 8000

CMD ["uvicorn", "churn_prediction.api:app", "--host", "0.0.0.0", "--port", "8000"]
