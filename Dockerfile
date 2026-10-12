FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY src/ /app/src/
COPY prompts/ /app/prompts/
COPY data/ /app/data/
COPY out/ /app/out/
CMD ["python","-m","src.cli","run","--split","dev"]
