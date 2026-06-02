import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import shutil
import os

PROCESSED_DIR = "data/processed"
TMP_DIR = Path("data/processed/tmp")

if TMP_DIR.exists():
    shutil.rmtree(TMP_DIR)

TMP_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Liquidity Sentinel", layout="wide", page_icon="🏦")

st.title("🏦 Liquidity Sentinel — система раннего предупреждения стресса ликвидности")

# Загрузка данных
@st.cache_data
def load_data():
    lsi_path = os.path.join(PROCESSED_DIR, "lsi_output.parquet")
    if os.path.exists(lsi_path):
        df = pd.read_parquet(lsi_path)
        return df
    return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("Данные не найдены. Запустите сначала модуль агрегации:")
    st.code("python -c \"from modules.lsi_aggregator import run; run()\"")
    st.stop()

# ---- Основная метрика ----
latest = df.iloc[-1]
lsi = latest['lsi']
status = latest['status']

status_colors = {"ЗЕЛЁНЫЙ": "green", "ЖЁЛТЫЙ": "orange", "КРАСНЫЙ": "red"}

col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1.5])

with col1:
    st.metric("📊 Текущий LSI", f"{lsi:.1f}")
    st.markdown(f"<h2 style='color:{status_colors[status]}'>Статус: {status}</h2>", unsafe_allow_html=True)

with col2:
    st.metric("📉 Мин LSI", f"{df['lsi'].min():.1f}")
    st.metric("📈 Макс LSI", f"{df['lsi'].max():.1f}")

with col3:
    st.metric("📊 Средний LSI", f"{df['lsi'].mean():.1f}")
    st.metric("🔴 Красных дней", f"{(df['lsi'] >= 70).sum()}")

with col4:
    st.metric("📅 Последний день", latest['date'].strftime('%Y-%m-%d'))
    st.metric("🟢 Зелёных дней", f"{(df['lsi'] < 40).sum()}")

# ---- График LSI ----
st.subheader("📈 Индекс стресса ликвидности (LSI)")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['date'], y=df['lsi'],
    mode='lines',
    name='LSI',
    line=dict(color='#1f77b4', width=2),
    fill='tozeroy',
    fillcolor='rgba(31,119,180,0.1)'
))

# Добавляем зоны
fig.add_hline(y=40, line_dash="dash", line_color="orange", 
              annotation_text="Жёлтая зона (40)", annotation_position="bottom right")
fig.add_hline(y=70, line_dash="dash", line_color="red", 
              annotation_text="Красная зона (70)", annotation_position="top right")

fig.update_layout(
    title="Динамика индекса стресса ликвидности за всё время",
    yaxis_title="LSI (0-100)",
    xaxis_title="Дата",
    height=500,
    hovermode='x unified'
)

st.plotly_chart(fig, width='stretch')

# ---- Сигналы модулей (последние 365 дней) ----
st.subheader("📊 Сигналы модулей (последние 365 дней)")

last_year = df.tail(365)

# Создаём 2 ряда по 2 графика для лучшего отображения
col1, col2 = st.columns(2)

with col1:
    if 'stress_m1' in df.columns:
        fig_m1 = go.Figure()
        fig_m1.add_trace(go.Scatter(
            x=last_year['date'], 
            y=last_year['stress_m1'],
            mode='lines', 
            name='M1 - резервы',
            line=dict(color='blue', width=2)
        ))
        fig_m1.update_layout(
            title="M1: Усреднение резервов",
            yaxis_title="Стресс (0-10)",
            height=300
        )
        st.plotly_chart(fig_m1, width='stretch')

with col2:
    if 'stress_m2' in df.columns:
        fig_m2 = go.Figure()
        fig_m2.add_trace(go.Scatter(
            x=last_year['date'], 
            y=last_year['stress_m2'],
            mode='lines', 
            name='M2 - репо ЦБ',
            line=dict(color='red', width=2)
        ))
        fig_m2.update_layout(
            title="M2: Аукционы репо ЦБ",
            yaxis_title="Стресс (0-10)",
            height=300
        )
        st.plotly_chart(fig_m2, width='stretch')

col3, col4 = st.columns(2)

with col3:
    if 'stress_m3' in df.columns:
        fig_m3 = go.Figure()
        fig_m3.add_trace(go.Scatter(
            x=last_year['date'], 
            y=last_year['stress_m3'],
            mode='lines', 
            name='M3 - ОФЗ',
            line=dict(color='green', width=2)
        ))
        fig_m3.update_layout(
            title="M3: Аукционы ОФЗ",
            yaxis_title="Стресс (0-10)",
            height=300
        )
        st.plotly_chart(fig_m3, width='stretch')

with col4:
    if 'stress_m5' in df.columns:
        fig_m5 = go.Figure()
        fig_m5.add_trace(go.Scatter(
            x=last_year['date'], 
            y=last_year['stress_m5'],
            mode='lines', 
            name='M5 - казначейство',
            line=dict(color='purple', width=2)
        ))
        fig_m5.update_layout(
            title="M5: Средства казначейства",
            yaxis_title="Стресс (0-10)",
            height=300
        )
        st.plotly_chart(fig_m5, width='stretch')

# ---- Дополнительная информация по M4 (налоговый мультипликатор) ----
if 'Seasonal_Factor' in df.columns and 'Tax_Week_Flag' in df.columns:
    st.subheader("📅 Налоговый календарь (M4)")
    
    # Показываем текущий налоговый фактор
    latest_factor = latest['Seasonal_Factor']
    tax_week = latest['Tax_Week_Flag']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Текущий налоговый фактор", f"{latest_factor:.2f}")
    with col2:
        st.metric("Налоговая неделя", "ДА" if tax_week == 1 else "НЕТ")
    
    # График налогового фактора за последние 90 дней
    last_90 = df.tail(90)
    fig_factor = go.Figure()
    fig_factor.add_trace(go.Scatter(
        x=last_90['date'], 
        y=last_90['Seasonal_Factor'],
        mode='lines',
        name='Seasonal Factor',
        line=dict(color='orange', width=2),
        fill='tozeroy',
        fillcolor='rgba(255,165,0,0.2)'
    ))
    fig_factor.update_layout(
        title="Налоговый мультипликатор (1.0-1.4)",
        yaxis_title="Коэффициент",
        xaxis_title="Дата",
        height=300
    )
    st.plotly_chart(fig_factor, width='stretch')

# ---- Алерт ----
st.subheader("🚨 Текущий алерт")

if lsi >= 70:
    st.error("🔴 **КРАСНЫЙ УРОВЕНЬ СТРЕССА (70-100)**\n\nВысокий риск дефицита ликвидности! Рекомендуется:\n- Усилить мониторинг рынка репо\n- Проверить прогнозы налоговых платежей\n- Подготовить план действий")
elif lsi >= 40:
    st.warning("🟡 **ЖЁЛТЫЙ УРОВЕНЬ СТРЕССА (40-70)**\n\nПовышенное внимание к ликвидности. Рекомендуется:\n- Следить за аукционами репо ЦБ\n- Мониторить размещения ОФЗ\n- Обратить внимание на движение средств казначейства")
else:
    st.success("🟢 **ЗЕЛЁНЫЙ УРОВЕНЬ (0-40)**\n\nЛиквидность в норме. Продолжать штатный мониторинг.")

# ---- Таблица с последними днями ----
st.subheader("📋 Последние 10 дней")

available_cols = ['date', 'lsi', 'status', 'stress_m1', 'stress_m2', 'stress_m3', 'stress_m5']
available_cols = [c for c in available_cols if c in df.columns]

display_df = df.tail(10)[available_cols].copy()
for col in display_df.select_dtypes(include=['float64', 'int64']).columns:
    display_df[col] = display_df[col].round(2)
st.dataframe(display_df, use_container_width=True)

# ---- Backtest: стресс-эпизоды ----
st.subheader("📊 Backtest: стресс-эпизоды (по данным ЦБ)")

episodes = {
    'Декабрь 2014': ('2014-12-01', '2014-12-31'),
    'Февраль-март 2022': ('2022-02-01', '2022-03-31'),
    'Август 2023': ('2023-08-01', '2023-08-31'),
}

episode_stats = []
for name, (start, end) in episodes.items():
    mask = (df['date'] >= start) & (df['date'] <= end)
    if mask.any():
        episode_df = df[mask]
        # Определяем статус по среднему LSI
        mean_lsi = episode_df['lsi'].mean()
        if mean_lsi >= 70:
            ep_status = "КРАСНЫЙ"
        elif mean_lsi >= 40:
            ep_status = "ЖЁЛТЫЙ"
        else:
            ep_status = "ЗЕЛЁНЫЙ"
        
        episode_stats.append({
            'Эпизод': name,
            'Средний LSI': f"{episode_df['lsi'].mean():.1f}",
            'Макс LSI': f"{episode_df['lsi'].max():.1f}",
            'Статус': ep_status
        })

if episode_stats:
    st.table(pd.DataFrame(episode_stats))
else:
    st.info("Данные за стресс-эпизоды не найдены в текущем диапазоне дат")

# ---- Распределение LSI по зонам ----
st.subheader("📊 Распределение LSI по зонам")

zone_counts = {
    'Зелёная (0-40)': (df['lsi'] < 40).sum(),
    'Жёлтая (40-70)': ((df['lsi'] >= 40) & (df['lsi'] < 70)).sum(),
    'Красная (70-100)': (df['lsi'] >= 70).sum()
}

fig_pie = go.Figure(data=[go.Pie(
    labels=list(zone_counts.keys()),
    values=list(zone_counts.values()),
    marker_colors=['green', 'orange', 'red'],
    hole=0.3
)])
fig_pie.update_layout(title="Процент времени в каждой зоне", height=400)
st.plotly_chart(fig_pie, width='stretch')

# ---- Информация о системе ----
st.subheader("ℹ️ О системе")

st.markdown("""
**Liquidity Sentinel** — система раннего предупреждения стресса ликвидности рублёвого денежного рынка.

**Компоненты:**
- **M1** — усреднение обязательных резервов (данные ЦБ)
- **M2** — аукционы репо ЦБ (7-дневные аукционы)
- **M3** — размещение ОФЗ (спрос/предложение)
- **M4** — налоговый календарь (сезонный мультипликатор)
- **M5** — средства федерального казначейства (ЕКС)

**Формула LSI:** взвешенная сумма сигналов с сигмоидой + налоговый мультипликатор

**Интерпретация:**
- 🟢 0-40: нормальный уровень, штатный режим
- 🟡 40-70: повышенное внимание, возможен стресс
- 🔴 70-100: высокий стресс, требуется немедленная реакция
""")

# ---- Footer ----
st.divider()
col1, col2 = st.columns(2)
with col1:
    st.caption(f"📅 Последнее обновление: {latest['date'].strftime('%Y-%m-%d')}")
with col2:
    st.caption(f"📊 Всего дней в истории: {len(df)} | Период: {df['date'].min().strftime('%Y-%m-%d')} — {df['date'].max().strftime('%Y-%m-%d')}")

st.caption("🏦 Система раннего предупреждения стресса ликвидности | ПСБ Казначейство")