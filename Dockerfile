FROM python:3.10-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir mindspore==2.2.10

COPY backend/ ./backend/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

EXPOSE 8000

CMD cd backend && uvicorn api.main:app --host 0.0.0.0 --port $PORT