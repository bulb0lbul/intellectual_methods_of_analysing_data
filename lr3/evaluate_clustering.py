import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, adjusted_rand_score, classification_report, confusion_matrix

FEAT_COLS = ["mean", "std", "range"]
CLASS_NAMES = {1: "Спокойное (1)", 2: "Буря (2)"}
COLORS = {1: "tab:blue", 2: "tab:red"}


def main():
    mem = pd.read_csv("memberships.csv")
    feats = pd.read_csv("features.csv")
    centers = pd.read_csv("centers.csv", index_col="cluster")

    y_true = mem["label"].to_numpy()
    y_pred = mem["cluster"].to_numpy()
    classes = [1, 2]
    U = mem[[f"u{c}" for c in classes]].to_numpy()

    # ---------- метрики ----------
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print("Матрица ошибок (строки - истинный класс, столбцы - кластер):")
    print(pd.DataFrame(cm, index=[CLASS_NAMES[c] for c in classes], columns=[CLASS_NAMES[c] for c in classes]).to_string())
    print(classification_report(y_true, y_pred, labels=classes,
                                target_names=[CLASS_NAMES[c] for c in classes], digits=3))
    print(f"Adjusted Rand Index: {adjusted_rand_score(y_true, y_pred):.3f}")
    print(f"Коэффициент разбиения FPC: {np.mean(np.sum(U ** 2, axis=1)):.3f} "
          f"(1 - чёткое разбиение, 1/Nc = {1 / len(classes):.2f} - полная размытость)")

    # ---------- ошибочно отнесённые сутки ----------
    wrong = mem[mem["label"] != mem["cluster"]]
    print(f"\nОшибочно отнесено суток: {len(wrong)} из {len(mem)}")
    if len(wrong):
        print(wrong[["date", "label", "cluster", "u1", "u2"]].round(3).to_string(index=False))
        print("Ошибки по годам:", wrong["date"].str[:4].value_counts().sort_index().to_dict())

    # ---------- 1. матрица ошибок ----------
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=[CLASS_NAMES[c] for c in classes]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Кластер (FCM)")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Матрица ошибок")
    fig.tight_layout()
    fig.savefig("plots/confusion_matrix.png", dpi=150)

    # ---------- 2. разделение на классы в пространстве признаков ----------
    pairs = [("mean", "std"), ("mean", "range"), ("std", "range")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for row, (title, labels_) in enumerate((("истинные классы", y_true), ("кластеры FCM", y_pred))):
        for col, (fx, fy) in enumerate(pairs):
            ax = axes[row, col]
            for c in classes:
                m = labels_ == c
                ax.scatter(feats.loc[m, fx], feats.loc[m, fy], s=28, alpha=0.8, color=COLORS[c], label=CLASS_NAMES[c])
            if row == 1:  # ошибочные точки выделяем, центры отмечаем
                ax.scatter(feats.loc[y_true != y_pred, fx], feats.loc[y_true != y_pred, fy], s=110, facecolors="none", edgecolors="k", linewidths=1.2, label="ошибка")
                ax.scatter(centers[fx], centers[fy], marker="X", s=180, color="gold", edgecolors="k", label="центры")
            ax.set_xlabel(fx)
            ax.set_ylabel(fy)
            ax.set_title(f"{fx} - {fy}: {title}")
            ax.grid(alpha=0.3)
            if col == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("plots/separation.png", dpi=150)

    # ---------- 3. распределение степеней принадлежности ----------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 21)
    for c in classes:
        ax.hist(mem.loc[mem["label"] == c, "u2"], bins=bins, alpha=0.6,
                color=COLORS[c], label=f"истинный класс: {CLASS_NAMES[c]}")
    ax.axvline(0.5, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Степень принадлежности кластеру 2 (буря), u2")
    ax.set_ylabel("Число суток")
    ax.set_title("Распределение степеней принадлежности")
    ax.legend()
    fig.tight_layout()
    fig.savefig("plots/membership_hist.png", dpi=150)

    print("\nСохранено: confusion_matrix.png, separation.png, membership_hist.png")


if __name__ == "__main__":
    main()