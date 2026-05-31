import requests
import pandas as pd
import re
import chardet
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import html

RAW_DIR = Path("data/raw")
FNS_OPENDATA_URL = "https://www.nalog.gov.ru/opendata/7707329152-kalendar/"

def get_all_xml_urls():
    """Возвращает список всех XML-файлов (актуальный + предыдущие релизы)."""
    print("  Загрузка страницы ФНС...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(FNS_OPENDATA_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    
    soup = BeautifulSoup(resp.text, 'html.parser')
    urls = []
    
    table = soup.find('table', class_='border_table')
    if table:
        for row in table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 3:
                if "Гиперссылка (URL) на набор" in cells[1].get_text(strip=True):
                    a = cells[2].find('a')
                    if a and a.get('href'):
                        urls.append(a['href'])
                if "предыдущие релизы" in cells[1].get_text(strip=True).lower():
                    for a in cells[2].find_all('a'):
                        if a.get('href') and '.xml' in a['href']:
                            urls.append(a['href'])
    
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '.xml' in href and 'data-20' in href:
            if href.startswith('/'):
                href = 'https://data.nalog.ru' + href
            urls.append(href)
    
    urls = list(set(urls))
    print(f"  Найдено XML-файлов: {len(urls)}")
    return urls

def fetch_xml_content(xml_url):
    """Скачивает XML и определяет его кодировку автоматически."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(xml_url, headers=headers, timeout=30)
    resp.raise_for_status()
    
    # Определяем кодировку по содержимому
    detected = chardet.detect(resp.content)
    encoding = detected.get('encoding', 'utf-8')
    # Корректировка для русских кодировок
    if encoding.lower() == 'windows-1252':
        encoding = 'windows-1251'
    
    try:
        text = resp.content.decode(encoding)
    except UnicodeDecodeError:
        # fallback
        text = resp.content.decode('utf-8', errors='replace')
    
    return text

def clean_cdata(raw):
    """Очищает CDATA, декодирует HTML-сущности и убирает теги."""
    text = re.sub(r'<!\[CDATA\[', '', raw)
    text = re.sub(r'\]\]>', '', text)
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_tax_type(text):
    """Извлекает короткое название налога из очищенного текста."""
    text_lower = text.lower()
    tax_map = {
        'ндс': 'НДС',
        'ндфл': 'НДФЛ',
        'налог на прибыль': 'Налог на прибыль',
        'налог на имущество': 'Налог на имущество',
        'страховые взносы': 'Страховые взносы',
        'ндпи': 'НДПИ',
        'акцизы': 'Акцизы',
        'енп': 'ЕНП',
        'транспортный налог': 'Транспортный налог',
        'земельный налог': 'Земельный налог',
        'водный налог': 'Водный налог',
        'торговый сбор': 'Торговый сбор',
        'туристический налог': 'Туристический налог',
        'нпд': 'НПД',
        'аусн': 'АУСН',
        'косвенные налоги': 'Косвенные налоги',
    }
    for key, val in tax_map.items():
        if key in text_lower:
            return val
    return text[:60]

def month_name_to_number(name):
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    return months.get(name.lower())

def parse_single_xml(xml_content, source_url):
    """Парсит один XML и возвращает список событий (дата + полное описание + тип)."""
    events = []
    
    pos = 0
    while True:
        year_start = xml_content.find('<year', pos)
        if year_start == -1:
            break
        year_end = xml_content.find('</year>', year_start)
        if year_end == -1:
            break
        
        year_block = xml_content[year_start:year_end + 7]
        year_match = re.search(r'index="(\d+)"', year_block)
        if not year_match:
            pos = year_end + 1
            continue
        year = int(year_match.group(1))
        
        month_pos = 0
        while True:
            month_start = year_block.find('<month', month_pos)
            if month_start == -1:
                break
            month_end = year_block.find('</month>', month_start)
            if month_end == -1:
                break
            
            month_block = year_block[month_start:month_end + 8]
            month_match = re.search(r'name="(\w+)"', month_block)
            if month_match:
                month_name = month_match.group(1)
                month_num = month_name_to_number(month_name)
                
                if month_num:
                    day_pos = 0
                    while True:
                        day_start = month_block.find('<day', day_pos)
                        if day_start == -1:
                            break
                        if 'type="event"' not in month_block[day_start:day_start + 50]:
                            day_pos = day_start + 1
                            continue
                        
                        day_end = month_block.find('</day>', day_start)
                        if day_end == -1:
                            break
                        
                        day_block = month_block[day_start:day_end + 6]
                        num_match = re.search(r'num="(\d+)"', day_block)
                        if num_match:
                            day_num = int(num_match.group(1))
                            
                            content_start = day_block.find('>') + 1
                            content_end = day_block.rfind('<')
                            if content_start < content_end:
                                raw_content = day_block[content_start:content_end]
                                full_text = clean_cdata(raw_content)
                                if full_text:
                                    try:
                                        date = datetime(year, month_num, day_num)
                                        tax_type = extract_tax_type(full_text)
                                        events.append({
                                            'date': date,
                                            'tax_event': full_text,
                                            'tax_type': tax_type
                                        })
                                    except Exception as e:
                                        print(f"    Ошибка даты {year}-{month_name}-{day_num}: {e}")
                        day_pos = day_end + 1
            month_pos = month_end + 1
        pos = year_end + 1
    
    return events

def fetch_tax_calendar():
    print("\n=== Загрузка налогового календаря ФНС ===")
    urls = get_all_xml_urls()
    
    all_events = []
    for url in urls:
        filename = url.split('/')[-1]
        print(f"  Обработка: {filename}")
        try:
            xml_content = fetch_xml_content(url)
            events = parse_single_xml(xml_content, url)
            print(f"    Найдено событий: {len(events)}")
            all_events.extend(events)
        except Exception as e:
            print(f"    Ошибка: {e}")
    
    if not all_events:
        raise RuntimeError("Не найдено событий ни в одном XML")
    
    df = pd.DataFrame(all_events)
    df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    print(f"\n  Всего уникальных событий: {len(df)}")
    return df

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_tax_calendar()
    
    output_path = RAW_DIR / "tax_calendar.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nСохранено {len(df)} событий в {output_path}")
    print(f"Период: {df['date'].min().date()} — {df['date'].max().date()}")
    print("\nПервые 10 событий:")
    print(df.head(10))
    
    print("\nУникальные типы налогов (первые 20):")
    unique_types = df['tax_type'].unique()
    for t in unique_types[:20]:
        print(f"  {t}")

if __name__ == "__main__":
    main()