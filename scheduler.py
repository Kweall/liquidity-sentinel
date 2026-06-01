import schedule
import time
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('data/pipeline.log'),
        logging.StreamHandler()
    ]
)

def run_module(name: str, module_path: str):
    import importlib
    try:
        logging.info(f"Запуск {name}...")
        module = importlib.import_module(module_path)
        module.run()
        logging.info(f"{name} завершён")
    except Exception as e:
        logging.error(f"{name} упал: {e}", exc_info=True)


def run_lsi():
    run_module("LSI", "modules.lsi_aggregator")

def job_m1_monthly():
    run_module("M1", "modules.m1_reserves")
    run_lsi()

def job_m2_daily():
    if datetime.now().weekday() < 5:
        run_module("M2", "modules.m2_repo")
        run_lsi()

def job_m3_twice_weekly():
    run_module("M3", "modules.m3_ofz")
    run_lsi()

def job_m4_weekly():
    run_module("M4", "modules.m4_taxes")
    run_lsi()

def job_m5_monthly():
    run_module("M5", "modules.m5_treasury")
    run_lsi()

def setup_schedule():
    schedule.every().day.at("21:00").do(job_m2_daily)
    schedule.every().tuesday.at("10:00").do(job_m3_twice_weekly)
    schedule.every().wednesday.at("10:00").do(job_m3_twice_weekly)
    schedule.every().day.at("09:00").do(
        lambda: job_m1_monthly() if datetime.now().day == 1 else None
    )
    schedule.every().day.at("09:30").do(
        lambda: job_m5_monthly() if datetime.now().day in (1, 15) else None
    )
    schedule.every().sunday.at("08:00").do(job_m4_weekly)

if __name__ == "__main__":
    logging.info("Scheduler запущен")
    logging.info("Расписание:")
    logging.info("  M1 — 1-е число каждого месяца в 09:00")
    logging.info("  M2 — каждый рабочий день в 21:00")
    logging.info("  M3 — вторник и среда в 10:00")
    logging.info("  M4 — каждое воскресенье в 08:00")
    logging.info("  M5 — 1-е и 15-е числа в 09:30")

    logging.info("Первичный запуск всех модулей...")
    for name, path in [
        ("M1", "modules.m1_reserves"),
        ("M2", "modules.m2_repo"),
        ("M3", "modules.m3_ofz"),
        ("M5", "modules.m5_treasury"),
        ("M4", "modules.m4_taxes"),
    ]:
        run_module(name, path)
    run_lsi()

    setup_schedule()

    while True:
        schedule.run_pending()
        time.sleep(60)