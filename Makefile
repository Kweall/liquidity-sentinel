.PHONY: help build run stop clean shell test

help:
	@echo "Доступные команды:"
	@echo "  make build    - Собрать Docker образ"
	@echo "  make run      - Запустить контейнер"
	@echo "  make stop     - Остановить контейнер"
	@echo "  make clean    - Очистить данные и остановить"
	@echo "  make shell    - Зайти в контейнер"
	@echo "  make test     - Запустить тесты"
	@echo "  make update   - Обновить данные"
	@echo "  make logs     - Посмотреть логи"

build:
	docker compose build

run:
	docker compose up -d
	@echo ""
	@echo "✅ Дашборд доступен по адресу: http://localhost:8501"
	@echo "📊 Страница аналитика: http://localhost:8501/Analyst"
	@echo "📈 SHAP анализ: http://localhost:8501/SHAP_Analysis"
	@echo ""
	@echo "Для просмотра логов: make logs"

stop:
	docker compose down

clean: stop
	rm -rf data/processed/*.parquet
	docker compose rm -f

shell:
	docker compose exec liquidity-sentinel /bin/bash

test:
	docker compose exec liquidity-sentinel python -c "from modules.lsi_aggregator_ml import run; run()"

update:
	docker compose exec liquidity-sentinel python -c "from modules.lsi_aggregator_ml import run; run()"

logs:
	docker compose logs -f

status:
	docker compose ps
