FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN playwright install --with-deps chromium

WORKDIR /app

RUN mkdir -p data/raw data/processed data/logs

COPY . .

EXPOSE 8501

CMD ["supervisord", "-c", "/app/supervisord.conf"]