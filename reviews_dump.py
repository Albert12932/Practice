import time
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

from serpapi import GoogleSearch
from dotenv import load_dotenv
import os

def get_api_key() -> str:
    load_dotenv()
    key = os.getenv("SERPAPI_API_KEY")
    if not key:
        raise RuntimeError("SERPAPI_API_KEY не найден. Создайте .env или задайте переменную окружения.")
    return key

def find_place_with_ludocid(query: str, api_key: str, lang: str = "ru", gl: str = "ru"):
    place_meta = {}

    local_params = {
        "engine": "google_local",
        "q": query,
        "hl": lang,
        "gl": gl,
        "api_key": api_key
    }
    local_res = GoogleSearch(local_params).get_dict()
    local_items = local_res.get("local_results", [])
    if local_items:
        best = local_items[0]
        place_meta["title"] = best.get("title") or best.get("name")
        place_meta["address"] = best.get("address")
        place_meta["ludocid"] = best.get("ludocid") or best.get("cid")
        place_meta["place_id_local"] = best.get("place_id")  # иногда тоже бывает

    # B) Google Maps search — чтобы получить data_id / place_id
    maps_params = {
        "engine": "google_maps",
        "q": query,
        "type": "search",
        "hl": lang,
        "api_key": api_key
    }
    maps_res = GoogleSearch(maps_params).get_dict()

    if "place_results" in maps_res and maps_res["place_results"]:
        pr = maps_res["place_results"]
        place_meta.setdefault("title", pr.get("title"))
        place_meta.setdefault("address", pr.get("address"))
        place_meta["place_id"] = pr.get("place_id")
        place_meta["data_id"] = pr.get("data_id")
        place_meta["data_cid"] = pr.get("data_cid") or pr.get("cid")
    elif "local_results" in maps_res and maps_res["local_results"]:
        lr0 = maps_res["local_results"][0]
        place_meta.setdefault("title", lr0.get("title"))
        place_meta.setdefault("address", lr0.get("address"))
        place_meta["place_id"] = lr0.get("place_id")
        place_meta["data_id"] = lr0.get("data_id")
        place_meta["data_cid"] = lr0.get("data_cid") or lr0.get("cid")

    return place_meta

def fetch_all_reviews_with_fallback(api_key: str, ids: dict, lang: str = "ru", sort_by: str = "newest",
                                    max_pages: int | None = None, sleep_sec: float = 2.0):
    """
    Пробуем три варианта идентификатора:
    1) data_id (наиболее рекомендуемый SerpAPI для отзывов)
    2) ludocid / data_cid (CID)
    3) place_id (как крайний случай)
    """
    tried = []
    for key in ["data_id", "ludocid", "data_cid", "place_id"]:
        val = ids.get(key)
        if not val:
            continue
        tried.append((key, val))

        params = {
            "engine": "google_maps_reviews",
            "hl": lang,
            "api_key": api_key,
            "sort_by": sort_by,
        }
        params[key] = val

        all_reviews = []
        next_token = None
        page = 0
        while True:
            if next_token:
                params["next_page_token"] = next_token
            res = GoogleSearch(params).get_dict()
            batch = res.get("reviews", [])
            all_reviews.extend(batch)

            pag = res.get("serpapi_pagination") or {}
            next_token = pag.get("next_page_token")
            page += 1
            if (not next_token) or (max_pages and page >= max_pages):
                break

            import time
            time.sleep(sleep_sec)

        if all_reviews:
            return all_reviews, key  # успех с этим типом ID

    return [], tried  # ничего не нашли, вернём список, чем пробовали

def normalize_review(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "review_id": r.get("review_id"),
        "rating": r.get("rating"),
        "date": r.get("date") or r.get("published_date"),
        "name": (r.get("user") or {}).get("name") or r.get("user_name"),
        "profile_url": (r.get("user") or {}).get("link") or r.get("user_url"),
        "local_guide": (r.get("user") or {}).get("is_local_guide"),
        "review_text": r.get("snippet") or r.get("text") or "",
        "likes": r.get("likes") or r.get("thumbs_up_count"),
        "images_count": (r.get("images") and len(r.get("images"))) or 0,
        "owner_response": (r.get("owner_response") or {}).get("text") if r.get("owner_response") else None,
        "owner_response_date": (r.get("owner_response") or {}).get("date") if r.get("owner_response") else None,
        "translated": r.get("is_translated"),
        "response_translated": (r.get("owner_response") or {}).get("is_translated") if r.get("owner_response") else None,
    }

def save_json_csv(place_meta: Dict[str, Any], reviews: List[Dict[str, Any]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON (сырой ответ)
    with open(out_dir / "reviews_raw.json", "w", encoding="utf-8") as f:
        json.dump({"place": place_meta, "reviews": reviews}, f, ensure_ascii=False, indent=2)

    # CSV (нормализовано)
    normalized = [normalize_review(r) for r in reviews]
    fieldnames = list(normalized[0].keys()) if normalized else [
        "review_id","rating","date","name","profile_url","local_guide",
        "review_text","likes","images_count","owner_response","owner_response_date",
        "translated","response_translated"
    ]
    with open(out_dir / "reviews.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in normalized:
            w.writerow(row)

def main():
    parser = argparse.ArgumentParser(description="Выгрузка всех отзывов Google Maps через SerpAPI")
    parser.add_argument("query", help='Запрос поиска: "Гостиница <Название>, <Город>"')
    parser.add_argument("--lang", default="ru", help="Язык интерфейса (hl), по умолчанию ru")
    parser.add_argument("--sort", default="newest", choices=["newest","most_relevant"], help="Сортировка отзывов")
    parser.add_argument("--max_pages", type=int, default=None, help="Ограничить число страниц (для тестов)")
    parser.add_argument("--out", default="out_reviews", help="Папка для результатов")
    args = parser.parse_args()

    api_key = get_api_key()
    place = find_place_with_ludocid(args.query, api_key, args.lang, "ru")
    print("IDs:", place)

    reviews, used = fetch_all_reviews_with_fallback(
        api_key=api_key,
        ids=place,
        lang=args.lang,
        sort_by=args.sort,
        max_pages=args.max_pages,
        sleep_sec=2.0
    )
    print(f"Собрано отзывов: {len(reviews)} (использован идентификатор: {used})")


    out_dir = Path(args.out)
    save_json_csv(place, reviews, out_dir)
    print(f"Готово! JSON и CSV сохранены в: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
