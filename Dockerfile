FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# CPU-only PyTorch; network access is required at Docker build time.
RUN python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt
COPY src/ /app/src/
COPY prompts/ /app/prompts/
COPY docs/ /app/docs/
COPY Makefile Dockerfile /app/
RUN mkdir -p /app/data/inputs /app/out/predictions /app/out/checkpoints /app/out/plans
CMD ["python", "-m", "src.submission", "run"]
