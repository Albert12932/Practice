import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter
from dateutil import parser as du_parser

import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

#NLP пакеты и стоп-слова
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Гарантируем стоп-слова
try:
    _ = stopwords.words("russian")
except LookupError:
    nltk.download("stopwords")

# Конфиг/пути
st.set_page_config(page_title="Аналитика отзывов", layout="wide")
DATA_PATH = Path(r"C:\Users\shoma\OneDrive\Рабочий стол\pract\out_reviews\dataset_reviews.csv")

# Вспомогательные функции
RU_MONTHS = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель", 5: "Май", 6: "Июнь",
    7: "Июль", 8: "Август", 9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
}

STOP_RU = set(stopwords.words("russian"))
CUSTOM_STOPS = {
    "гостиница","отель","номер","номера","отзыв","отзывы","очень","приехали","спасибо",
    "были","был","была","это","всё","все","так","ещё","нам","нас","вам","есть",
    "просто","будто","типа","вообще","могли","мог","могла","будем","своих","свой","свои",
}
STOP_ALL = STOP_RU | CUSTOM_STOPS

RED_STEMS = [
    "грязн", "шум", "холод", "жар", "запах", "вон",
    "таракан", "плесен", "сырост", "кондиционер", "отоплен",
    "плох", "долго", "медлен", "уборк", "простын", "полотен",
    "интернет", "wifi", "вайф", "парковк", "ресепшен", "заселен", "задерж"
]

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^а-яёa-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_series(series: pd.Series):
    toks = []
    for t in series.fillna("").astype(str).map(clean_text):
        toks.extend([w for w in t.split() if w not in STOP_ALL and len(w) >= 3])
    return toks

def top_bigrams(series: pd.Series, topk=10):
    corpus = series.fillna("").astype(str).map(clean_text).tolist()
    cv = CountVectorizer(
        ngram_range=(2,2),
        token_pattern=r"(?u)\b[а-яёa-z]{3,}\b",
        stop_words=list(STOP_ALL)
    )
    if not any(len(x.split()) >= 2 for x in corpus):
        return []
    X = cv.fit_transform(corpus)
    if X.shape[1] == 0:
        return []
    freqs = X.sum(axis=0).A1
    vocab = cv.get_feature_names_out()
    pairs = list(zip(vocab, freqs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:topk]

def has_red_issue(text: str) -> bool:
    t = clean_text(text)
    return any(stem in t for stem in RED_STEMS)

def ensure_datetime_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Убеждаемся, что есть Год/Месяц и ключевой datetime для группировок."""
    out = df.copy()
    if "Год" not in out.columns or "Месяц" not in out.columns:
        date_col = None
        for c in ["Дата_ISO", "date_iso", "date", "published_date", "publishedDate", "time", "publishedTime"]:
            if c in out.columns:
                date_col = c; break
        if date_col is not None:
            dt = pd.to_datetime(out[date_col], errors="coerce")
            out["Год"] = dt.dt.year
            out["Месяц"] = dt.dt.month
    out["_key"] = pd.to_datetime(dict(year=out["Год"], month=out["Месяц"], day=1), errors="coerce")
    return out

def sentiment_from_rating(r):
    try:
        r = float(r)
    except:
        return None
    if r >= 4: return "Позитивные"
    if r <= 2: return "Негативные"
    return "Нейтральные"

# Загрузка данных
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
TEXT_COL = None
for c in ["Текст отзыва", "text", "review_text", "snippet"]:
    if c in df.columns:
        TEXT_COL = c; break
if TEXT_COL is None:
    st.error("Не найдена колонка с текстом отзыва.")
    st.stop()

df = ensure_datetime_cols(df)
df["Рейтинг"] = pd.to_numeric(df["Рейтинг"], errors="coerce")

# Боковая панель (фильтры)
st.sidebar.header("Фильтры")
years = sorted([int(y) for y in df["Год"].dropna().unique()])
year_sel = st.sidebar.multiselect("Год", options=years, default=years)
months = list(range(1,13))
month_labels = [RU_MONTHS[m] for m in months]
month_map = dict(zip(month_labels, months))
month_sel_labels = st.sidebar.multiselect("Месяц", options=month_labels, default=month_labels)
month_sel = [month_map[m] for m in month_sel_labels]

# Диапазон рейтинга (на всякий)
rmin, rmax = float(np.nanmin(df["Рейтинг"])), float(np.nanmax(df["Рейтинг"]))
rsel = st.sidebar.slider("Фильтр по рейтингу", min_value=1.0, max_value=5.0, value=(1.0, 5.0), step=0.5)

# Фильтрация
mask = (
    df["Год"].isin(year_sel) &
    df["Месяц"].isin(month_sel) &
    df["Рейтинг"].between(rsel[0], rsel[1], inclusive="both")
)
dff = df.loc[mask].copy()

st.title("Аналитика отзывов по гостинице")
st.caption("Интерактивный дашборд: фильтрация в левой панели, вкладки ниже.")

# Вкладки
tab1, tab2 = st.tabs(["📊 Обзор", "📝 Текстовый анализ"])

#Обзорные графики (коротко)
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        # Средняя оценка по месяцам (объединяя годы)
        monthly_avg = (
            dff.dropna(subset=["Месяц","Рейтинг"])
               .astype({"Месяц":"int"})
               .groupby("Месяц")["Рейтинг"]
               .mean()
               .reindex(range(1,13))
        )
        fig = px.bar(
            monthly_avg.reset_index().rename(columns={"index":"Месяц", "Рейтинг":"Средний рейтинг"}),
            x="Месяц", y="Средний рейтинг",
            title="Средняя оценка по месяцам (все выбранные годы)"
        )
        fig.update_xaxes(tickmode="array", tickvals=list(range(1,13)), ticktext=[RU_MONTHS[m] for m in range(1,13)])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Количество отзывов по годам
        by_year = (
            dff.dropna(subset=["Год"])
               .astype({"Год":"int"})
               .groupby("Год").size()
               .reset_index(name="Количество")
               .sort_values("Год")
        )
        fig = px.bar(by_year, x="Год", y="Количество", title="Количество отзывов по годам")
        st.plotly_chart(fig, use_container_width=True)

    # Пай-чарт: позитив/нейтрал/негатив
    cats = dff["Рейтинг"].apply(sentiment_from_rating)
    counts = pd.Series(cats).value_counts().reindex(["Позитивные","Нейтральные","Негативные"]).fillna(0)
    fig = px.pie(values=counts.values, names=counts.index, title="Структура отзывов")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    colA, colB = st.columns(2)

    #Распределение рейтингов
    with colA:
        if dff["Рейтинг"].notna().sum() > 0:
            fig = px.histogram(dff, x="Рейтинг", nbins=5, title="Распределение рейтингов (1–5)")
            fig.update_layout(xaxis=dict(tickmode="array", tickvals=[1,2,3,4,5]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных для распределения рейтингов.")

    #Длина отзывов
    with colB:
        dff["Длина_слов"] = dff[TEXT_COL].fillna("").apply(lambda s: len(str(s).split()))
        bins = [0, 25, 50, 100, 10_000]
        labels = ["0–25", "26–50", "51–100", "101+"]

        dff["Категория длины"] = pd.cut(dff["Длина_слов"], bins=bins, labels=labels, right=True, include_lowest=True)
        counts = dff["Категория длины"].value_counts().reindex(labels, fill_value=0).reset_index()
        counts.columns = ["Диапазон", "Количество"]

        fig = px.bar(counts, x="Диапазон", y="Количество", title="Длина отзывов (в словах)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Просмотр отзывов (первые 200)")
    st.dataframe(dff[[ "Рейтинг", "Год", "Месяц", TEXT_COL ]].head(200))

#Текстовый анализ
with tab2:
    st.subheader("Облака слов (позитив ≥4★ / негатив ≤2★)")
    colL, colR = st.columns(2)

    pos_df = dff[dff["Рейтинг"] >= 4]
    neg_df = dff[dff["Рейтинг"] <= 2]

    def texts_to_corpus(series: pd.Series) -> str:
        cleaned = series.fillna("").astype(str).map(clean_text)
        tokens = []
        for t in cleaned:
            tokens.extend([w for w in t.split() if w not in STOP_ALL and len(w) > 2])
        return " ".join(tokens)

    with colL:
        corpus_pos = texts_to_corpus(pos_df[TEXT_COL])
        if len(corpus_pos) > 0:
            wc_pos = WordCloud(width=900, height=540, background_color="white", collocations=False, prefer_horizontal=0.9).generate(corpus_pos)
            fig_wc1, ax1 = plt.subplots(figsize=(8,5))
            ax1.imshow(wc_pos, interpolation="bilinear")
            ax1.axis("off")
            st.pyplot(fig_wc1, use_container_width=True)
        else:
            st.info("Недостаточно данных для позитивного облака.")

    with colR:
        corpus_neg = texts_to_corpus(neg_df[TEXT_COL])
        if len(corpus_neg) > 0:
            wc_neg = WordCloud(width=900, height=540, background_color="white", collocations=False, prefer_horizontal=0.9).generate(corpus_neg)
            fig_wc2, ax2 = plt.subplots(figsize=(8,5))
            ax2.imshow(wc_neg, interpolation="bilinear")
            ax2.axis("off")
            st.pyplot(fig_wc2, use_container_width=True)
        else:
            st.info("Недостаточно данных для негативного облака.")

    st.markdown("---")
    st.subheader("ТОП-слова и биграммы")

    topk = st.slider("TOP-N для частотных списков/биграмм", min_value=5, max_value=30, value=10, step=1)

    #ТОП-слова (позитив/негатив)
    col1, col2 = st.columns(2)
    with col1:
        pos_tokens = tokenize_series(pos_df[TEXT_COL])
        pos_top = Counter(pos_tokens).most_common(topk)
        if pos_top:
            words, counts = zip(*pos_top)
            fig = px.bar(x=list(words), y=list(counts), title=f"ТОП-{topk} слов — позитивные")
            fig.update_layout(xaxis_title="Слова", yaxis_title="Частота")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Недостаточно данных для ТОП-слов (позитив).")

    with col2:
        neg_tokens = tokenize_series(neg_df[TEXT_COL])
        neg_top = Counter(neg_tokens).most_common(topk)
        if neg_top:
            words, counts = zip(*neg_top)
            fig = px.bar(x=list(words), y=list(counts), title=f"ТОП-{topk} слов — негативные")
            fig.update_layout(xaxis_title="Слова", yaxis_title="Частота")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Недостаточно данных для ТОП-слов (негатив).")

    #ТОП-биграмм (позитив/негатив)
    col3, col4 = st.columns(2)
    with col3:
        pos_bi = top_bigrams(pos_df[TEXT_COL], topk=topk)
        if pos_bi:
            labels, vals = zip(*pos_bi)
            fig = px.bar(x=list(labels), y=list(vals), title=f"ТОП-{topk} биграмм — позитивные")
            fig.update_layout(xaxis_title="Биграммы", yaxis_title="Частота")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Недостаточно данных для биграмм (позитив).")

    with col4:
        neg_bi = top_bigrams(neg_df[TEXT_COL], topk=topk)
        if neg_bi:
            labels, vals = zip(*neg_bi)
            fig = px.bar(x=list(labels), y=list(vals), title=f"ТОП-{topk} биграмм — негативные")
            fig.update_layout(xaxis_title="Биграммы", yaxis_title="Частота")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Недостаточно данных для биграмм (негатив).")
