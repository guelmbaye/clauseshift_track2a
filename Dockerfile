FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY src/ /app/src/
COPY prompts/ /app/prompts/
COPY docs/ /app/docs/
COPY Makefile Dockerfile /app/
RUN mkdir -p /app/data/inputs /app/out/predictions /app/out/checkpoints
CMD ["python", "-m", "src.submission", "run"]
