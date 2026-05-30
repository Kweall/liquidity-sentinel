import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

PROCESSED_DIR = "data/processed"

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
fig.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Жёлтая зона (40)", annotation_position="bottom right")
fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Красная зона (70)", annotation_position="top right")

fig.update_layout(
    title="Динамика индекса стресса ликвидности за всё время",
    yaxis_title="LSI (0-100)",
    xaxis_title="Дата",
    height=500,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# ---- Графики модулей (теперь 3 графика) ----
st.subheader("📊 Сигналы модулей (последние 365 дней)")

last_year = df.tail(365)

col1, col2, col3 = st.columns(3)

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

# ---- Алерт ----
st.subheader("🚨 Текущий алерт")

if lsi >= 70:
    st.error("🔴 **КРАСНЫЙ УРОВЕНЬ СТРЕССА (70-100)**\n\nВысокий риск дефицита ликвидности! Рекомендуется:\n- Усилить мониторинг рынка репо\n- Проверить прогнозы налоговых платежей\n- Подготовить план действий")
elif lsi >= 40:
    st.warning("🟡 **ЖЁЛТЫЙ УРОВЕНЬ СТРЕССА (40-70)**\n\nПовышенное внимание к ликвидности. Рекомендуется:\n- Следить за аукционами репо ЦБ\n- Мониторить размещения ОФЗ")
else:
    st.success("🟢 **ЗЕЛЁНЫЙ УРОВЕНЬ (0-40)**\n\nЛиквидность в норме. Продолжать штатный мониторинг.")

# ---- Таблица с последними днями ----
st.subheader("📋 Последние 10 дней")

available_cols = [c for c in ['date', 'lsi', 'status', 'stress_m1', 'stress_m2', 'stress_m3'] if c in df.columns]
display_df = df.tail(10)[available_cols].copy()
for col in display_df.select_dtypes(include=['float64', 'int64']).columns:
    display_df[col] = display_df[col].round(2)
st.dataframe(display_df)

# ---- Backtest: стресс-эпизоды ----
st.subheader("📊 Backtest: стресс-эпизоды")

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
        episode_stats.append({
            'Эпизод': name,
            'Средний LSI': f"{episode_df['lsi'].mean():.1f}",
            'Макс LSI': f"{episode_df['lsi'].max():.1f}",
            'Статус': episode_df['status'].mode().iloc[0] if not episode_df['status'].mode().empty else 'Н/Д'
        })

if episode_stats:
    st.table(pd.DataFrame(episode_stats))
else:
    st.info("Данные за стресс-эпизоды не найдены в текущем диапазоне дат")

# ---- Информация ----
st.caption(f"📅 Последнее обновление: {latest['date'].strftime('%Y-%m-%d')} | Система раннего предупреждения стресса ликвидности")
st.caption(f"📊 Всего дней в истории: {len(df)} | Период: {df['date'].min().strftime('%Y-%m-%d')} — {df['date'].max().strftime('%Y-%m-%d')}")