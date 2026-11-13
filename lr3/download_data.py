import csv
import datetime
import io
import math
import time
from pathlib import Path

import requests

URL = "https://cosmicrays.oulu.fi/webform/onlinequery.cgi"
STATION = "OULU"
COLUMN = "CorrectedCountRate[cts/min]"  # поправка на давление и эффективность
OUT = Path("data")

CLASS1 = [
    (2018, 1, 3), (2018, 1, 5), (2018, 1, 6), (2018, 1, 7), (2018, 1, 10),
    (2018, 1, 11), (2018, 1, 16), (2018, 1, 17), (2018, 2, 3), (2018, 2, 11),
    (2018, 2, 13), (2018, 3, 12), (2018, 3, 13), (2018, 4, 3), (2018, 4, 6),
    (2018, 4, 16), (2018, 4, 17), (2018, 5, 1), (2018, 5, 25),
    (2018, 6, 9), (2018, 6, 10), (2018, 6, 11), (2018, 6, 12), (2018, 6, 15),
    (2018, 6, 16), (2018, 6, 29), (2018, 7, 1), (2018, 7, 2), (2018, 9, 20),
    (2018, 10, 17), (2018, 10, 18), (2018, 10, 19), (2018, 10, 24), (2018, 10, 27),
    (2018, 10, 28), (2018, 10, 29), (2018, 10, 30), (2018, 11, 22), (2018, 11, 23),
    (2018, 12, 12), (2018, 12, 13), (2018, 12, 14), (2018, 12, 15), (2018, 12, 24),
]

CLASS2 = [
    (2018, 1, 1), (2018, 1, 13), (2018, 3, 18), (2018, 4, 12), (2018, 4, 20),
    (2018, 5, 5), (2018, 5, 6), (2018, 5, 7), (2018, 6, 1), (2018, 6, 18),
    (2018, 6, 26), (2018, 7, 5), (2018, 8, 20), (2018, 9, 10), (2018, 9, 11),
    (2018, 9, 14), (2018, 9, 22), (2018, 10, 8), (2018, 10, 10), (2018, 10, 19),
    (2018, 11, 5), (2018, 12, 8), (2018, 12, 26), (2018, 12, 28), (2018, 12, 29),
    (2019, 1, 5), (2019, 1, 25), (2019, 2, 1), (2019, 2, 28), (2019, 3, 1),
    (2019, 3, 17), (2019, 5, 14), (2019, 8, 5), (2019, 8, 31), (2019, 9, 1),
    (2019, 10, 25), (2020, 3, 4), (2020, 3, 5), (2020, 6, 8), (2020, 9, 28),
    (2020, 10, 25), (2021, 3, 1), (2021, 5, 12), (2021, 10, 12), (2021, 11, 4),
]


def fetch_day(y, m, d):
    # Форма принимает время только с шагом 30 минут (максимум 23:30), поэтому
    # просим до 00:00 следующих суток, а последнюю строку потом отбрасываем.
    nxt = datetime.date(y, m, d) + datetime.timedelta(days=1)
    params = {
        "station": STATION,
        "startday": f"{d:02d}", "startmonth": f"{m:02d}", "startyear": y,
        "starttime": "00:00",
        "endday": f"{nxt.day:02d}", "endmonth": f"{nxt.month:02d}", "endyear": nxt.year,
        "endtime": "00:00",
        "resolution": 1,
        "outputmode": "csv",
    }
    r = requests.get(URL, params=params, timeout=60)
    r.raise_for_status()
    reader = csv.DictReader(io.StringIO(r.text))
    if reader.fieldnames is None or COLUMN not in reader.fieldnames:
        raise ValueError(f"нет столбца {COLUMN}; получено: {reader.fieldnames}")
    rows = []
    for rec in reader:
        raw = (rec.get(COLUMN) or "").strip()
        try:
            v = float(raw)
            val = "" if math.isnan(v) else f"{v / 60:.4f}"  # cts/min -> counts/s
        except ValueError:
            val = ""  # пропуск
        rows.append((rec["Timestamp"], val))
    # убрать последнюю строку (00:00 следующих суток), оставить только сами сутки
    day_prefix = f"{y}-{m:02d}-{d:02d}T"
    return [row for row in rows if row[0].startswith(day_prefix)]


def main():
    for cls, dates in (("class1", CLASS1), ("class2", CLASS2)):
        folder = OUT / cls
        folder.mkdir(parents=True, exist_ok=True)
        for y, m, d in sorted(set(dates)):
            path = folder / f"{y}-{m:02d}-{d:02d}.csv"
            if path.exists():
                continue
            rows = None
            for attempt in range(3):
                try:
                    rows = fetch_day(y, m, d)
                    break
                except (requests.RequestException, ValueError) as e:
                    print(f"retry {attempt + 1}: {path.name}: {e}")
                    time.sleep(5)
            if not rows:
                print(f"FAIL {path.name}")
                continue
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write("time,value\n")
                for t, v in rows:
                    f.write(f"{t},{v}\n")
            gaps = sum(1 for _, v in rows if v == "")
            note = "" if len(rows) == 1440 else "  <-- ожидалось 1440"
            print(f"{path.name}: {len(rows)} строк, пропусков {gaps}{note}")
            time.sleep(1)


if __name__ == "__main__":
    main()