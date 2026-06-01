#!/bin/bash
set -e

echo "=========================================="
echo "Liquidity Sentinel - система раннего"
echo "предупреждения стресса ликвидности"
echo "=========================================="
echo ""

# Ждём, пока Ollama запустится
echo "Ожидание запуска Ollama..."
while ! curl -s http://ollama:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo "✅ Ollama готов"

# Проверяем, есть ли модель qwen2.5:3b
echo "Проверка модели qwen2.5:3b..."
if curl -s http://ollama:11434/api/tags | grep -q "qwen2.5:3b"; then
    echo "✅ Модель уже есть"
else
    echo "Модель qwen2.5:3b не найдена. Скачиваем и создаём..."
    
    # Скачиваем GGUF файл во временную директорию
    cd /tmp
    if [ ! -f "qwen2.5-3b.gguf" ]; then
        echo "Загрузка qwen2.5-3b-instruct-q4_k_m.gguf (2GB)..."
        wget -O qwen2.5-3b.gguf "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
    fi
    
    # Создаём модель через Ollama API
    echo "Создание модели qwen2.5:3b через Ollama API..."
    
    # Сначала создаём Modelfile
    cat > Modelfile << 'MODEL_EOF'
FROM /tmp/qwen2.5-3b.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9

TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}{{ range .Messages }}{{ if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{ end }}{{ end }}<|im_start|>assistant
"""
MODEL_EOF
    
    # Используем curl для создания модели через API
    # Сначала нужно скопировать файл модели в контейнер Ollama
    echo "Копирование файла модели в Ollama контейнер..."
    docker cp /tmp/qwen2.5-3b.gguf liquidity-ollama:/tmp/qwen2.5-3b.gguf
    docker cp /tmp/Modelfile liquidity-ollama:/tmp/Modelfile
    
    # Создаём модель в Ollama контейнере
    echo "Создание модели в Ollama..."
    docker exec liquidity-ollama ollama create qwen2.5:3b -f /tmp/Modelfile
    
    # Очистка
    docker exec liquidity-ollama rm -f /tmp/qwen2.5-3b.gguf /tmp/Modelfile
    rm -f /tmp/qwen2.5-3b.gguf /tmp/Modelfile
    
    echo "✅ Модель qwen2.5:3b успешно создана"
fi

# Проверяем, есть ли уже данные, если нет — запускаем агрегатор
if [ ! -f "data/processed/lsi_output_ml.parquet" ]; then
    echo "Первичная загрузка данных и расчёт LSI..."
    python -c "from modules.lsi_aggregator_ml import run; run()"
else
    echo "Данные уже загружены."
fi

# Запускаем Streamlit
echo ""
echo "=========================================="
echo "Запуск дашборда: http://localhost:8501"
echo "=========================================="
echo ""

streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
