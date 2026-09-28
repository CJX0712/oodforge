FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# 默认运行端到端基准并落盘 benchmark.json
CMD ["python", "-m", "oodforge.examples.run_demo", "--out", "benchmark.json"]
