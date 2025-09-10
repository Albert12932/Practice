import json, re, numpy as np, pandas as pd
from datetime import datetime
from dateutil import parser as du_parser
import dateparser
from pathlib import Path

# Пути
base = Path(r"C:\Users\shoma\OneDrive\Рабочий стол\pract\out_reviews")
json_path = base / "reviews_raw.json"
out_csv  = base / "dataset_reviews.csv"
out_xlsx = base / "dataset_reviews.xlsx"

def parse_rating_to_float(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    m = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)\s*", s)
    if m:
        num = float(m.group(1).replace(",", "."))
        den = float(m.group(2).replace(",", "."))
        return (num/den)*10.0 if den>0 else np.nan
    s = s.replace(",", ".")
    try: return float(s)
    except: return np.nan

def normalize_rating_rule(v):
    if pd.isna(v): return np.nan
    if v > 5:
        if 6 <= v <= 7:  return 4.0
        if 8 <= v <= 10: return 5.0
        return 5.0
    return float(v)

def try_parse_iso(s):
    try:
        return du_parser.parse(str(s))
    except:
        return None

def get_abs_datetime(review, relative_base=None):
    """
    1) Пытаемся найти абсолютную дату в разных полях
    2) Если нет — парсим относительную 'date' с заданной базой
    """
    for k in ["timestamp", "unix_time", "time", "time_ms", "epoch", "date_ms", "date_seconds"]:
        v = review.get(k)
        if isinstance(v, (int, float)):
            try:
                ts = float(v)
                if ts > 1e12: ts = ts/1000.0
                return datetime.fromtimestamp(ts)
            except:
                pass

    for k in ["published_date", "publishedDate", "published_time",
              "publishedTime", "date_iso", "iso_date", "datetime", "date_utc"]:
        v = review.get(k)
        if v:
            dt = try_parse_iso(v)
            if dt: return dt

    rel = review.get("date")
    if rel:
        dt = dateparser.parse(
            str(rel),
            languages=["ru", "en"],
            settings={"RELATIVE_BASE": relative_base or datetime.now(),
                      "PREFER_DAY_OF_MONTH": "first"}
        )
        if dt: return dt

    return None

data = json.loads(json_path.read_text(encoding="utf-8"))
reviews = data.get("reviews", [])

relative_base = None

rows = []
for r in reviews:
    dt = get_abs_datetime(r, relative_base=relative_base)
    rating_raw  = parse_rating_to_float(r.get("rating"))
    rating_norm = normalize_rating_rule(rating_raw)
    text = r.get("snippet") or r.get("text") or ""
    author = (r.get("user") or {}).get("name") or r.get("user_name")

    rows.append({
        "Рейтинг": rating_norm,
        "Год": dt.year if dt else None,
        "Месяц": dt.month if dt else None,
        "Источник": "Google Maps",
        "Текст отзыва": str(text).strip(),
        "Автор": author,
        "Дата_ISO": dt.isoformat() if dt else None
    })

dataset = pd.DataFrame(rows)

dataset.to_csv(out_csv, index=False, encoding="utf-8-sig")
try: dataset.to_excel(out_xlsx, index=False)
except: pass

print("Готово:", out_csv)
print(dataset[["Год","Месяц"]].value_counts().sort_index())
