# 文件名: backend/Dockerfile, 版本号: 2.1
FROM python:3.10-slim

WORKDIR /app

ENV NLTK_DATA=/app/nltk_data
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV HF_ENDPOINT=https://hf-mirror.com

# --- 【核心修正】新增 libmagic1，这是文件类型识别的核心系统库 ---
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    curl \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libmagic1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*



COPY backend/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt


COPY backend/ .
COPY frontend/ /app/frontend

COPY backend/entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]