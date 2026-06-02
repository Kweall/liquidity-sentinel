import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import json
from pathlib import Path
from datetime import datetime, timedelta

PROCESSED_DIR = "data/processed"
CHAT_HISTORY_FILE = Path("data/processed/tmp/chat_history.json")
DEFAULT_MESSAGE = """👋 Добро пожаловать в аналитический модуль! Я помогу вам разобраться с ликвидностью.

Вот что я умею:

- 📊 Анализировать текущий LSI и его компоненты
- 📅 Отвечать на вопросы о конкретных датах и периодах
- 🔍 Находить стресс-эпизоды в истории
- 📈 Прогнозировать влияние налогов и аукционов ОФЗ

Задайте вопрос, например:

- Что происходило с ликвидностью в марте 2022?
- Почему в августе 2023 вырос LSI?
- Покажи периоды максимального стресса за последний год
"""

st.set_page_config(page_title="Аналитик — Liquidity Sentinel", layout="wide", page_icon="🧠")

st.title("🧠 Аналитик — интеллектуальный помощник по ликвидности")
st.markdown("Задавайте вопросы о текущей ситуации, исторических стрессах или прогнозах ликвидности.")

# =========================
# Загрузка всех данных системы
# =========================
@st.cache_data
def load_system_data():
    data = {}
    for name in ['lsi_output', 'm1_output', 'm2_output', 'm3_output', 'm4_output', 'm5_output']:
        path = os.path.join(PROCESSED_DIR, f"{name}.parquet")
        if os.path.exists(path):
            data[name] = pd.read_parquet(path)
        else:
            data[name] = pd.DataFrame()
    return data

def load_chat_history():
    if CHAT_HISTORY_FILE.exists():
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return [
        {
            "role": "assistant",
            "content": DEFAULT_MESSAGE
        }
    ]


def save_chat_history(messages):
    CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

@st.cache_data
def load_tax_calendar():
    path = "data/raw/tax_calendar.csv"
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=['date'])
    return pd.DataFrame()

@st.cache_data
def load_ofz_auctions():
    path = "data/raw/ofz_clean.csv"
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=['date'])
    return pd.DataFrame()

data = load_system_data()
tax_df = load_tax_calendar()
ofz_df = load_ofz_auctions()
lsi_df = data.get('lsi_output', pd.DataFrame())

# Описание модулей для контекста
MODULES_DESCRIPTION = """
Модуль M1: Усреднение обязательных резервов. Сигнал о том, что банки держат больше нормы — признак стресса.
Модуль M2: Аукционы репо ЦБ. Высокий спрос на репо — признак дефицита ликвидности.
Модуль M3: Аукционы ОФЗ. Низкий спрос на госдолг — признак нехватки свободных денег.
Модуль M4: Налоговый календарь. Сезонные оттоки ликвидности в налоговые периоды.
Модуль M5: Средства казначейства. Оттоки бюджета создают давление на ликвидность.
"""

# =========================
# Функции для поиска релевантного контекста
# =========================
def find_relevant_context(question):
    """Ищет релевантные данные по вопросу пользователя"""
    context_parts = []
    question_lower = question.lower()
    
    # Поиск по конкретным датам
    import re
    date_pattern = r'(\d{4}-\d{2}-\d{2})|(\d{2}\.\d{2}\.\d{4})|(март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь|январь|февраль)\s*(\d{4})?'
    dates_found = re.findall(date_pattern, question_lower)
    
    if dates_found and not lsi_df.empty:
        # Пытаемся найти конкретную дату
        for date_match in dates_found:
            date_str = date_match[0] or date_match[1]
            if date_str:
                try:
                    target_date = pd.to_datetime(date_str, dayfirst=True)
                    week_data = lsi_df[(lsi_df['date'] >= target_date - timedelta(days=3)) & 
                                       (lsi_df['date'] <= target_date + timedelta(days=3))]
                    if not week_data.empty:
                        context_parts.append(f"Данные около {target_date.strftime('%Y-%m-%d')}:")
                        for _, row in week_data.iterrows():
                            context_parts.append(f"  LSI {row['lsi']:.1f} ({row['status']})")
                except:
                    pass
    
    # Поиск по месяцам/годам
    months = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь']
    for month in months:
        if month in question_lower and not lsi_df.empty:
            # Находим год
            year_match = re.search(r'(20\d{2})', question_lower)
            year = int(year_match.group(1)) if year_match else 2022
            month_num = months.index(month) + 1
            month_data = lsi_df[(lsi_df['date'].dt.year == year) & (lsi_df['date'].dt.month == month_num)]
            if not month_data.empty:
                context_parts.append(f"Данные за {month} {year}:")
                context_parts.append(f"  Средний LSI: {month_data['lsi'].mean():.1f}")
                context_parts.append(f"  Макс LSI: {month_data['lsi'].max():.1f}")
                context_parts.append(f"  Статус: {month_data['status'].mode().iloc[0] if not month_data['status'].mode().empty else 'Н/Д'}")
    
    # Стресс-эпизоды
    if 'стресс' in question_lower or 'кризис' in question_lower:
        episodes = {
            'декабрь 2014': ('2014-12-01', '2014-12-31'),
            'февраль-март 2022': ('2022-02-01', '2022-03-31'),
            'август 2023': ('2023-08-01', '2023-08-31'),
        }
        context_parts.append("\nИСТОРИЧЕСКИЕ СТРЕСС-ЭПИЗОДЫ:")
        for name, (start, end) in episodes.items():
            mask = (lsi_df['date'] >= start) & (lsi_df['date'] <= end)
            if mask.any():
                ep_data = lsi_df[mask]
                context_parts.append(f"{name}: средний LSI {ep_data['lsi'].mean():.1f}, макс {ep_data['lsi'].max():.1f}, статус {ep_data['status'].mode().iloc[0]}")
    
    # Текущая ситуация
    if 'сейчас' in question_lower or 'текущ' in question_lower or 'сегодня' in question_lower:
        if not lsi_df.empty:
            latest = lsi_df.iloc[-1]
            context_parts.append(f"\nТЕКУЩАЯ СИТУАЦИЯ (на {latest['date'].strftime('%Y-%m-%d')}):")
            context_parts.append(f"  LSI: {latest['lsi']:.1f} ({latest['status']})")
            if 'stress_m1' in latest:
                context_parts.append(f"  M1 стресс: {latest['stress_m1']:.2f}")
            if 'stress_m2' in latest:
                context_parts.append(f"  M2 стресс: {latest['stress_m2']:.2f}")
            if 'stress_m3' in latest:
                context_parts.append(f"  M3 стресс: {latest['stress_m3']:.2f}")
            if 'stress_m5' in latest:
                context_parts.append(f"  M5 стресс: {latest['stress_m5']:.2f}")
    
    # Налоговые даты
    if 'налог' in question_lower and not tax_df.empty:
        upcoming = tax_df[tax_df['date'] > datetime.now()].head(5)
        if not upcoming.empty:
            context_parts.append("\nБЛИЖАЙШИЕ НАЛОГОВЫЕ ДАТЫ:")
            for _, row in upcoming.iterrows():
                context_parts.append(f"  {row['date'].strftime('%Y-%m-%d')}: {row.get('tax_type', 'налоговое событие')}")
    
    # Аукционы ОФЗ
    if 'офз' in question_lower and not ofz_df.empty:
        try:
            ofz_dates = pd.to_datetime(ofz_df['date'])
            upcoming_ofz = ofz_dates[ofz_dates > datetime.now()].head(5)
            if not upcoming_ofz.empty:
                context_parts.append("\nБЛИЖАЙШИЕ АУКЦИОНЫ ОФЗ:")
                for d in upcoming_ofz:
                    context_parts.append(f"  {d.strftime('%Y-%m-%d')}")
        except:
            pass
    
    return "\n".join(context_parts) if context_parts else "Нет дополнительных данных по этому запросу."

# =========================
# Функция для RAG-ответа
# =========================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

def get_rag_response(question, chat_history):
    """RAG-ответ с поиском релевантных данных"""
    
    # 1. Находим релевантный контекст
    relevant_context = find_relevant_context(question)
    
    # 2. История диалога
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-5:]])
    
    # 3. Статистика по модулям
    stats_text = ""
    if not lsi_df.empty:
        stats_text = f"""
ОБЩАЯ СТАТИСТИКА:
- Всего дней в истории: {len(lsi_df)}
- Красных дней (LSI ≥ 70): {(lsi_df['lsi'] >= 70).sum()}
- Жёлтых дней (40-70): {((lsi_df['lsi'] >= 40) & (lsi_df['lsi'] < 70)).sum()}
- Зелёных дней (<40): {(lsi_df['lsi'] < 40).sum()}
- Максимальный LSI: {lsi_df['lsi'].max():.1f}
- Минимальный LSI: {lsi_df['lsi'].min():.1f}
"""
    
    # 4. Формируем промпт
    prompt = f"""Ты — аналитический помощник по системе раннего предупреждения стресса ликвидности.
Отвечай на вопросы пользователя, используя ТОЛЬКО предоставленные данные.

ОПИСАНИЕ МОДУЛЕЙ:
{MODULES_DESCRIPTION}

{stats_text}

{relevant_context}

ИСТОРИЯ ДИАЛОГА:
{history_text}

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}

ПРАВИЛА ОТВЕТА:
1. Используй ТОЛЬКО данные из предоставленного контекста.
2. Если данных нет, скажи: "В предоставленных данных нет информации об этом."
3. Отвечай на русском языке, кратко (2-4 предложения).
4. Если пользователь просит конкретную дату или период — найди в контексте.
5. Не выдумывай цифры и даты.

ОТВЕТ:
"""
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "max_tokens": 500}
            },
            timeout=45
        )
        if response.status_code == 200:
            return response.json().get('response', "Не удалось получить ответ.").strip()
        else:
            return f"Ошибка LLM: {response.status_code}"
    except Exception as e:
        return f"Ошибка подключения к Ollama: {e}\nУбедитесь, что Ollama запущен (ollama serve)"

# =========================
# Интерфейс чата
# =========================
if "analyst_messages" not in st.session_state:
    st.session_state.analyst_messages = load_chat_history()

for message in st.session_state.analyst_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Задайте вопрос о ликвидности..."):
    st.session_state.analyst_messages.append({"role": "user", "content": prompt})
    save_chat_history(st.session_state.analyst_messages)
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Анализирую данные системы..."):
            answer = get_rag_response(prompt, st.session_state.analyst_messages)
            st.markdown(answer)
            st.session_state.analyst_messages.append({"role": "assistant", "content": answer})
            save_chat_history(st.session_state.analyst_messages)

# =========================
# Боковая панель с информацией
# =========================
with st.sidebar:
    st.markdown("### 📊 О системе")
    if not lsi_df.empty:
        latest = lsi_df.iloc[-1]
        st.metric("Текущий LSI", f"{latest['lsi']:.1f}")
        st.metric("Красных дней (LSI ≥ 70)", f"{(lsi_df['lsi'] >= 70).sum()}")
        st.metric("Жёлтых дней (40-70)", f"{((lsi_df['lsi'] >= 40) & (lsi_df['lsi'] < 70)).sum()}")
        st.metric("Зелёных дней (<40)", f"{(lsi_df['lsi'] < 40).sum()}")
    
    st.markdown("---")
    st.markdown("### 🔍 Примеры вопросов")
    st.info("""
    - Что происходило с ликвидностью в марте 2022?
    - Почему в августе 2023 вырос LSI?
    - Покажи периоды максимального стресса
    - Как налоговая неделя влияет на LSI?
    - Что сейчас с ликвидностью?
    """)
    st.markdown("---")
    if st.button("🗑️ Очистить историю чата"):
        st.session_state.analyst_messages = [
            {
                "role": "assistant",
                "content": DEFAULT_MESSAGE
            }
        ]

        save_chat_history(st.session_state.analyst_messages)

        st.rerun()
