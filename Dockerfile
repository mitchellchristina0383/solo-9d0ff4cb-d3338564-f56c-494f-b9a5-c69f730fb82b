FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV SOURCE_DIR=/data/sample/source \
    OUTPUT_ROOT=/data/outputs

EXPOSE 8000

# --factory: app.main.create_app() builds the app without import-time
# filesystem side effects. Single worker on purpose: job state lives in
# this process (see README).
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
