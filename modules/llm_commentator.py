import requests
import json
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

def build_prompt(lsi_value, status, modules_contrib, active_flags, tax_calendar, upcoming_ofz):
    prompt = f"""Ты — аналитик по ликвидности денежного рынка. 
Проанализируй текущую ситуацию и напиши короткий комментарий (3-5 предложений).

Текущий LSI: {lsi_value:.1f} (шкала 0-100, где 0-40 зелёный, 40-70 жёлтый, 70-100 красный)
Статус: {status}

Вклад модулей в сигнал (нормированный от 0 до 100, где 100 — максимальный вклад):
- Усреднение резервов (M1) — отражает перестраховку банков: {modules_contrib.get('m1', 0):.1f}%
- Аукционы РЕПО ЦБ (M2) — спрос на экстренное фондирование: {modules_contrib.get('m2', 0):.1f}%
- Аукционы ОФЗ (M3) — аппетит к госдолгу: {modules_contrib.get('m3', 0):.1f}%
- Казначейство (M5) — оттоки/притоки бюджета: {modules_contrib.get('m5', 0):.1f}%

Активные флаги: {', '.join(active_flags) if active_flags else 'нет'}

Ближайшие налоговые даты: {', '.join(tax_calendar[:3]) if tax_calendar else 'неизвестны'}
Ближайшие аукционы ОФЗ: {', '.join(upcoming_ofz[:3]) if upcoming_ofz else 'неизвестны'}

Напиши комментарий на русском языке в 3-5 предложений. Используй профессиональный, но понятный язык. Не пиши вступлений типа "я аналитик", сразу переходи к сути.
"""
    return prompt

def generate_commentary(prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "max_tokens": 500,
                    "top_p": 0.9,
                    "stop": ["\n\n", "```"]
                }
            },
            timeout=60
        )
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            return f"Ошибка LLM: статус {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "Не удалось подключиться к Ollama. Убедитесь, что Ollama запущен (ollama serve)."
    except Exception as e:
        return f"Ошибка при вызове LLM: {e}"

def add_commentary_to_lsi(lsi_value, status, modules_contrib, active_flags, tax_calendar, upcoming_ofz):
    """Основная функция для генерации комментария"""
    prompt = build_prompt(lsi_value, status, modules_contrib, active_flags, tax_calendar, upcoming_ofz)
    return generate_commentary(prompt)