FROM python:3.11-slim

# Системные зависимости (если нужны)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip
RUN pip install --no-cache-dir --upgrade pip

# Копируем и устанавливаем Python зависимости
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Рабочая директория
WORKDIR /workspace