FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || pip install --no-cache-dir numpy scipy scikit-learn pytest

COPY . .

# 默认运行端到端基准演示
CMD ["python", "-m", "oodforge.examples.run_demo"]
