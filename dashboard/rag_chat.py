import streamlit as st
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Загрузка модели эмбеддингов
@st.cache_resource
def load_embedder():
    return SentenceTransformer('intfloat/multilingual-e5-small')

# Загрузка данных системы
@st.cache_data
def load_system_data():
    lsi_df = pd.read_parquet("data/processed/lsi_output.parquet")
    m1_df = pd.read_parquet("data/processed/m1_output.parquet")
    m2_df = pd.read_parquet("data/processed/m2_output.parquet")
    m3_df = pd.read_parquet("data/processed/m3_output.parquet")
    m4_df = pd.read_parquet("data/processed/m4_output.parquet")
    m5_df = pd.read_parquet("data/processed/m5_output.parquet")
    return lsi_df, m1_df, m2_df, m3_df, m4_df, m5_df

# Построение векторного индекса
def build_vector_index(df, embedder):
    texts = df.apply(lambda row: f"Дата {row['date']}, LSI {row['lsi']:.1f}, статус {row['status']}", axis=1).tolist()
    embeddings = embedder.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return index, texts

# Ответ на вопрос
def ask_question(question, embedder, index, texts):
    query_embedding = embedder.encode([question])
    D, I = index.search(np.array(query_embedding).astype('float32'), k=3)
    context = "\n".join([texts[i] for i in I[0]])
    
    prompt = f"""Ты — аналитик по ликвидности. Используй контекст для ответа.
Контекст: {context}
Вопрос: {question}
Ответ:"""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen2.5:3b", "prompt": prompt, "stream": False},
        timeout=30
    )
    return response.json().get('response', "Не удалось получить ответ")