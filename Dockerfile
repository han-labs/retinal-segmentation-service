FROM python:3.11-slim

WORKDIR /app

# OpenCV cần vài thư viện hệ thống (bản headless là tốt nhất)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# EXPOSE không bắt buộc trên Render, có cũng được
EXPOSE 8000

# QUAN TRỌNG: dùng biến $PORT của Render (không fix 8000)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
