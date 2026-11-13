import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state

FEAT_COLS = ["mean", "std", "range"]

# Параметры метода (аналог options в MATLAB)
NC = 2          # число кластеров (в данных два известных класса)
M = 2.0         # показатель нечёткости
TOL = 1e-4      # порог остановки (изменение центров между итерациями)
MAXITER = 2000
SEED = 42


class FuzzyCMeans(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters: int, m: float = 2.0, tol: float = 1e-4, maxiter: int = 2000, random_state: int = 42, verbose: bool = False):
        self.n_clusters = n_clusters
        self.m = m
        self.tol = tol
        self.maxiter = maxiter
        self.random_state = random_state
        self.verbose = verbose

    def _compute_membership(self, dist: np.ndarray) -> np.ndarray:
        u = np.zeros(shape=(dist.shape[0], self.n_clusters))
        power = 2 / (self.m - 1)
        for k in range(dist.shape[0]):
            zero_mask = dist[k] == 0  # наблюдение совпало с центром

            if zero_mask.any():
                u[k] = zero_mask / zero_mask.sum()
            else:
                for i in range(self.n_clusters):
                    u[k, i] = 1 / np.sum((dist[k, i] / dist[k]) ** power)
        return u

    def fit(self, X: np.ndarray, y=None):
        X = check_array(X, dtype=np.float64)
        rng = check_random_state(self.random_state)

        # инициализация: случайная матрица принадлежностей, строки суммируются в 1
        u = rng.random_sample(size=(X.shape[0], self.n_clusters))  # N x k
        u /= u.sum(axis=1, keepdims=True)

        it = 0
        prev_cluster_centers = None
        cluster_centers = None
        while it < self.maxiter:
            w = u ** self.m  # N x k

            prev_cluster_centers = cluster_centers
            cluster_centers = w.T @ X / w.sum(axis=0)[:, None]  # (k x N) @ (N x d) = k x d

            dist = pairwise_distances(X, cluster_centers)
            u = self._compute_membership(dist)

            it += 1

            if it > 1:
                delta = np.linalg.norm(prev_cluster_centers - cluster_centers)
                if self.verbose:
                    print(f"Iter №{it} | delta {delta:.6f}")
                if delta < self.tol:
                    break

        self.cluster_centers_ = cluster_centers
        self.n_iter_ = it
        self.membership_ = u
        self.labels_ = np.argmax(u, axis=1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_membership(X), axis=1)

    def predict_membership(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["cluster_centers_"])
        X = check_array(X, dtype=np.float64)
        dist = pairwise_distances(X, self.cluster_centers_)
        return self._compute_membership(dist)


def main():
    df_std = pd.read_csv("features_std.csv")
    df_raw = pd.read_csv("features.csv")
    data = df_std[FEAT_COLS].to_numpy()  # N x D: строки - сутки

    fcm = FuzzyCMeans(n_clusters=NC, m=M, tol=TOL, maxiter=MAXITER, random_state=SEED).fit(data)

    # Согласование номеров: кластер -> класс (1 или 2) по максимальному совпадению с метками.
    # Метки используются только для именования кластеров, на сам результат кластеризации не влияют.
    classes = np.sort(df_std["label"].unique())  # [1, 2]
    assert len(classes) == NC, "число кластеров должно совпадать с числом классов"
    conf = np.array([[np.sum((fcm.labels_ == k) & (df_std["label"].to_numpy() == c)) for c in classes] for k in range(NC)])  # строки - кластеры, столбцы - классы
    rows, cols = linear_sum_assignment(-conf)  # оптимальное сопоставление
    order = rows[np.argsort(cols)]  # order[j] = исходный кластер, соответствующий классу classes[j]

    u = fcm.membership_[:, order]          # столбец j теперь соответствует классу classes[j]
    centers_std = fcm.cluster_centers_[order]
    labels = classes[np.argmax(u, axis=1)]  # чёткий кластер в нумерации классов (1, 2)

    # центры в физических единицах (обратное преобразование StandardScaler)
    scaler = StandardScaler().fit(df_raw[FEAT_COLS])
    cntr_phys = scaler.inverse_transform(centers_std)

    centers = pd.concat([pd.DataFrame(centers_std, columns=[f"{c}_std" for c in FEAT_COLS]), pd.DataFrame(cntr_phys, columns=FEAT_COLS)],axis=1)
    centers.index = classes
    centers.index.name = "cluster"
    centers.to_csv("centers.csv", float_format="%.6f")

    out = df_std[["date", "label"]].copy()
    for j, c in enumerate(classes):
        out[f"u{c}"] = u[:, j]
    out["cluster"] = labels
    out.to_csv("memberships.csv", index=False, float_format="%.6f")

    print(f"Nc={NC}, m={M}, итераций: {fcm.n_iter_} (maxiter={MAXITER})")
    print("Соответствие исходных кластеров классам:", {int(classes[j]): int(order[j]) for j in range(NC)})
    print("Центры кластеров (физические единицы):")
    print(centers[FEAT_COLS].round(3).to_string())
    print("Суток в кластерах:", dict(zip(*np.unique(labels, return_counts=True))))
    print("Сумма принадлежностей по суткам равна 1:", np.allclose(u.sum(axis=1), 1))


if __name__ == "__main__":
    main()