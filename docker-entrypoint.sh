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
echo "Проверка модели $LLM_MODEL..."
if ! curl -s http://ollama:11434/api/tags | grep -q "$LLM_MODEL"; then
    echo "Модель $LLM_MODEL не найдена. Скачивание (это займёт ~5-10 минут)..."
    curl -X POST http://ollama:11434/api/pull -d "{\"name\": \"$LLM_MODEL\"}"
    echo "✅ Модель загружена"
else
    echo "✅ Модель уже есть"
fi

# Проверяем, есть ли уже данные, если нет — запускаем агрегатор
if [ ! -f "data/processed/lsi_output_ml.parquet" ]; then
    echo "Первичная загрузка данных и расчёт LSI..."
    python -c "from modules.lsi_aggregator_ml import run; run()"
else
    echo "Данные уже загружены. Для обновления запустите: docker compose exec liquidity-sentinel python -c \"from modules.lsi_aggregator_ml import run; run()\""
fi

# Запускаем Streamlit
echo ""
echo "=========================================="
echo "Запуск дашборда: http://localhost:8501"
echo "=========================================="
echo ""

streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
