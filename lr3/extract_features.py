from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SRC = Path("fixed_data")
LABELS = {"class1": 1, "class2": 2}
FEAT_COLS = ["mean", "std", "range"]


def features(x: np.ndarray) -> dict:
    return {
        "mean": x.mean(),               # среднее значение (фоновый уровень)
        "std": x.std(ddof=1),           # стандартное отклонение (разброс, шум)
        "range": x.max() - x.min(),     # амплитуда (размах): x_max - x_min
    }


def main():
    # 1. Признаки как есть
    rows = []
    for cls, label in LABELS.items():
        for path in sorted((SRC / cls).glob("*.csv")):
            x = pd.read_csv(path)["value"].to_numpy(dtype=float)
            rows.append({"date": path.stem, "label": label, **features(x)})

    df = pd.DataFrame(rows)
    df.to_csv("features.csv", index=False, float_format="%.6f")

    print("Матрица data:", df[FEAT_COLS].shape, "(суток x признаков)")
    print(df.groupby("label").size().rename("суток в классе").to_string())
    print("Средние признаков по классам (до стандартизации):")
    print(df.groupby("label")[FEAT_COLS].mean().round(3).to_string())

    # 2. Стандартизация отдельным шагом (StandardScaler) -> новый файл
    scaler = StandardScaler()
    df_std = df.copy()
    df_std[FEAT_COLS] = scaler.fit_transform(df[FEAT_COLS])
    df_std.to_csv("features_std.csv", index=False, float_format="%.6f")

    print("Проверка стандартизации (среднее ~0, СКО ~1):")
    print(df_std[FEAT_COLS].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()