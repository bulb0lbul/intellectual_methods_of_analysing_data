from pathlib import Path

import pandas as pd

SRC = Path("data")
DST = Path("fixed_data")
EXPECTED_LEN = 1440


def main():
    total_files = 0
    total_gaps = 0
    for cls in ("class1", "class2"):
        (DST / cls).mkdir(parents=True, exist_ok=True)
        for path in sorted((SRC / cls).glob("*.csv")):
            df = pd.read_csv(path, dtype={"value": str})  # читаем как текст, чтобы видеть исходную запись
            original = df["value"].copy()
            df["value"] = pd.to_numeric(df["value"], errors="coerce")  # null/нечисловое -> NaN
            df.loc[df["value"] == 0, "value"] = float("nan")           # 0 тоже считается пропуском
            missing = df["value"].isna()
            gaps = int(missing.sum())

            if gaps:
                median = df["value"].median()  # медиана ряда (пропуски не учитываются)
                for idx in df.index[missing]:
                    old = original[idx]
                    old = "пусто" if pd.isna(old) or str(old).strip() == "" else old
                    print(f"{cls}/{path.name}: строка {idx + 2} ({df.at[idx, 'time']}): "
                          f"{old} -> {median:.4f}")
                df["value"] = df["value"].fillna(median)

            if len(df) != EXPECTED_LEN:
                print(f"ВНИМАНИЕ {cls}/{path.name}: {len(df)} строк вместо {EXPECTED_LEN}")

            df.to_csv(DST / cls / path.name, index=False, float_format="%.4f")
            total_files += 1
            total_gaps += gaps
            if gaps:
                print(f"{cls}/{path.name}: заполнено пропусков {gaps}")

    print(f"Готово: файлов {total_files}, пропусков заполнено {total_gaps}")


if __name__ == "__main__":
    main()