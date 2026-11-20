from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
except ModuleNotFoundError as error:
    raise SystemExit(
        "scikit-learn is not installed. Install it with:\n"
        "python3 -m pip install -r Concepts/regression/requirements.txt"
    ) from error


RANDOM_SEED = 7


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray
    y: np.ndarray


def true_response(x: np.ndarray) -> np.ndarray:
    return 2.0 + 1.2 * x + 1.8 * np.maximum(x, 0.0) ** 2


def draw_population(rng: np.random.Generator, n_rows: int) -> Dataset:
    x = rng.uniform(-3.0, 3.0, size=n_rows)
    noise = rng.normal(0.0, 1.5, size=n_rows)
    y = true_response(x) + noise
    return Dataset(x=x.reshape(-1, 1), y=y)


def sample_selection_probability(x: np.ndarray) -> np.ndarray:
    flat_x = x.ravel()
    return np.where(flat_x < 0.0, 0.90, 0.18)


def draw_biased_labeled_sample(rng: np.random.Generator, population: Dataset) -> Dataset:
    keep_probability = sample_selection_probability(population.x)
    selected = rng.random(population.x.shape[0]) < keep_probability
    return Dataset(x=population.x[selected], y=population.y[selected])


def assign_cluster(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    cluster = np.digitize(x.ravel(), edges[1:-1], right=False)
    return np.clip(cluster, 0, len(edges) - 2)


def estimate_cluster_weights(
    biased_x: np.ndarray,
    target_x: np.ndarray,
    n_clusters: int = 8,
) -> np.ndarray:
    edges = np.quantile(target_x.ravel(), np.linspace(0.0, 1.0, n_clusters + 1))
    target_cluster = assign_cluster(target_x, edges)
    biased_cluster = assign_cluster(biased_x, edges)

    target_counts = np.bincount(target_cluster, minlength=n_clusters)
    biased_counts = np.bincount(biased_cluster, minlength=n_clusters)

    target_share = target_counts / target_counts.sum()
    biased_share = np.maximum(biased_counts / biased_counts.sum(), 1e-12)
    cluster_weight = target_share / biased_share

    weights = cluster_weight[biased_cluster]
    return weights / weights.mean()


def fit_and_score(
    labeled: Dataset,
    test: Dataset,
    sample_weight: np.ndarray | None = None,
) -> tuple[LinearRegression, float]:
    model = LinearRegression()
    model.fit(labeled.x, labeled.y, sample_weight=sample_weight)
    predictions = model.predict(test.x)
    rmse = math.sqrt(mean_squared_error(test.y, predictions))
    return model, rmse


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    population = draw_population(rng, 10_000)
    target_unlabeled = draw_population(rng, 4_000)
    test = draw_population(rng, 4_000)
    labeled = draw_biased_labeled_sample(rng, population)

    weights = estimate_cluster_weights(labeled.x, target_unlabeled.x)

    unweighted_model, unweighted_rmse = fit_and_score(labeled, test)
    weighted_model, weighted_rmse = fit_and_score(labeled, test, sample_weight=weights)

    left_share = np.mean(labeled.x.ravel() < 0.0)
    target_left_share = np.mean(target_unlabeled.x.ravel() < 0.0)

    print("Cluster-based sample-selection bias correction")
    print(f"Labeled sample rows: {len(labeled.y)}")
    print(f"Target/unbiased x < 0 share: {target_left_share:.1%}")
    print(f"Biased labeled x < 0 share: {left_share:.1%}")
    print()
    print("Model fitted as: y_hat = intercept + slope * x")
    print(
        "Unweighted: "
        f"intercept={unweighted_model.intercept_:6.3f}, "
        f"slope={unweighted_model.coef_[0]:6.3f}, "
        f"target RMSE={unweighted_rmse:6.3f}"
    )
    print(
        "Weighted:   "
        f"intercept={weighted_model.intercept_:6.3f}, "
        f"slope={weighted_model.coef_[0]:6.3f}, "
        f"target RMSE={weighted_rmse:6.3f}"
    )
    print()
    print(
        "The weighted model uses inverse cluster-frequency weights estimated "
        "from the biased labeled sample and an unlabeled target sample."
    )


if __name__ == "__main__":
    main()
