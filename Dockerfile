FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY data ./data
COPY solution.py .

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import json,urllib.request,sys; sys.exit(0 if json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/health'))['status']=='ok' else 1)"

CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
