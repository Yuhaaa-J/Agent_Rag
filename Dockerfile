FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/var/hf

WORKDIR /app

# uv 比 pip 快一个数量级，直接装进系统环境，镜像更简单
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache .

COPY evals ./evals
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "agent_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
