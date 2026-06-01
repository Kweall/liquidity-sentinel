FROM python:3.11-slim

# Устанавливаем системные зависимости (включая docker CLI для копирования файлов)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    gnupg \
    zstd \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Рабочая директория
WORKDIR /app

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Создаём директории для данных
RUN mkdir -p data/raw data/processed models

# Открываем порты
EXPOSE 8501

# Скрипт для запуска всех сервисов
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
