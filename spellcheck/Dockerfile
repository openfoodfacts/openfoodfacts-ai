FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PYTHONPATH="/app/src"

WORKDIR /app

COPY ./models/model /app/model

COPY ./src /app/src

COPY ./scripts/batch/ /app

# Install required packages
COPY ./scripts/batch/requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint to the batch job script
ENTRYPOINT ["python main.py"]