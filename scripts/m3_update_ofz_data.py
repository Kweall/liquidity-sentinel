from pathlib import Path
import os
import requests
import shutil
import zipfile
import pandas as pd
from playwright.sync_api import sync_playwright

from .m3_data_preparation import build_ofz_clean_csv

BASE = Path(__file__).resolve().parent.parent   # корень проекта
RAW_DIR = BASE / "data" / "raw"
TMP_DIR = BASE / "data" / "tmp_ofz"


BASE_URL = "https://minfin.gov.ru/ru/perfomance/public_debt/internal/operations/ofz/auction"
SECTION_TITLE = "Таблицы по результатам проведения аукционов"
STOP_MARKER = "tablitsa_auktsionov_svodnaya_na_23.12.2015"

MOEX_FILES = [
    ("https://moex.com/iss/downloads/engines/stock/zcyc/prices.csv.zip", "prices.csv"),
    ("https://moex.com/iss/downloads/engines/stock/zcyc/dynamic.csv.zip", "dynamic.csv"),
]


def download_file(url, path: Path):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=60)
    if r.status_code in (403, 429, 503):
        raise RuntimeError(f"Blocked: {url}")
    r.raise_for_status()
    path.write_bytes(r.content)


def download_zip_and_extract(url, out_name: str):
    zip_path = TMP_DIR / f"{out_name}.zip"
    download_file(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)
    zip_path.unlink(missing_ok=True)
    print(f"extracted: {out_name}")


def download_moex():
    print("DOWNLOADING MOEX...")
    for url, name in MOEX_FILES:
        download_zip_and_extract(url, name)
    print("MOEX READY")


def get_ofz_links():
    links = set()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(BASE_URL, timeout=60000)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)

        headers = page.query_selector_all("h2, h3, h4")
        section = None
        for h in headers:
            try:
                if SECTION_TITLE.lower() in h.inner_text().lower():
                    section = h.evaluate_handle("el => el.closest('div, section')")
                    break
            except:
                continue

        if not section:
            raise RuntimeError("Section not found")
        print("SECTION LOCKED")

        while True:
            anchors = section.query_selector_all("a")
            stop = False
            for a in anchors:
                href = a.get_attribute("href")
                if not href:
                    continue
                if STOP_MARKER in href:
                    stop = True
                if ".xls" in href.lower():
                    if href.startswith("/"):
                        href = "https://minfin.gov.ru" + href
                    links.add(href)
            if stop:
                break
            btn = section.query_selector("a.button_more")
            if not btn:
                break
            try:
                btn.scroll_into_view_if_needed()
                btn.click()
                page.wait_for_timeout(800)
            except:
                break
        browser.close()
    return sorted(links)

def download_ofz_files(links):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for url in links:
        name = url.split("/")[-1]
        path = TMP_DIR / name
        print("downloading:", name)
        try:
            download_file(url, path)
            files.append(path)
        except Exception as e:
            print("failed:", name, e)
    return files

def main():
    print("START UPDATE OFZ DATA")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    links = get_ofz_links()
    print("FOUND OFZ LINKS:", len(links))

    ofz_files = download_ofz_files(links)
    print("DOWNLOADED OFZ FILES:", len(ofz_files))

    output_csv = RAW_DIR / "ofz_clean.csv"
    build_ofz_clean_csv(ofz_files, output_csv)

    download_moex()

    shutil.rmtree(TMP_DIR, ignore_errors=True)

    print("\n✅ Данные для модуля M3 обновлены:")
    print(f"   {output_csv}")
    print(f"   {RAW_DIR / 'dynamic.csv'}")
    print(f"   {RAW_DIR / 'prices.csv'}")


if __name__ == "__main__":
    main()