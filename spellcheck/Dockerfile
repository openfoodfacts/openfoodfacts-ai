FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PYTHONPATH="/app/src"

WORKDIR /app

COPY ./src /app

COPY ./scripts/batch/. /app

RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint to the batch job script
ENTRYPOINT ["python", "main.py"]