#!/bin/sh

# 检查是否需要等待Ollama服务
if [ "$WAIT_FOR_OLLAMA" = "true" ]; then
  OLLAMA_URL="http://ollama:11434"
  echo "WAIT_FOR_OLLAMA is true. Waiting for Ollama to be ready at ${OLLAMA_URL}..."

  # 循环检查Ollama服务，直到它返回成功的HTTP状态码
  until curl -sSf --max-time 5 "${OLLAMA_URL}" > /dev/null 2>&1; do
      echo "Ollama is not ready yet, sleeping for 5 seconds..."
      sleep 5
  done

  echo "Ollama is ready!"
else
  echo "WAIT_FOR_OLLAMA is not true. Skipping Ollama wait check."
fi

echo "Starting backend service..."

# 执行传递给这个脚本的原始命令 (即 "uvicorn app:app ...")
exec "$@"