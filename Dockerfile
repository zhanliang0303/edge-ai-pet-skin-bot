FROM python:3.11-slim

WORKDIR /app

# 安裝 Python 套件
COPY requirements_bot.txt .
RUN pip install --no-cache-dir -r requirements_bot.txt

# 複製檔案
COPY bot_mvp.py .
COPY docker_best.onnx .
COPY knowledge_base.json .
COPY .env .

# 啟動機器人
CMD ["python", "bot_mvp.py"]
