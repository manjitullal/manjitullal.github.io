"""Generate 1M rows x 10 columns of dummy ads data, saved as CSV and Parquet.

Also writes a small advertisers dimension table (5k rows) for the join benchmark.
"""
from pathlib import Path

import numpy as np
import pandas as pd

N_ROWS = 1_000_000
N_ADVERTISERS = 5_000
DATA_DIR = Path(__file__).parent / "data"


def make_events(n: int, rng: np.random.Generator) -> pd.DataFrame:
    impressions = rng.poisson(20, n) + 1
    return pd.DataFrame({
        "user_id":       rng.integers(0, 200_000, n),
        "ad_id":         rng.integers(0, 50_000, n),
        "advertiser_id": rng.integers(0, N_ADVERTISERS, n),
        "surface":       rng.choice(["homefeed", "search", "related_pins", "shopping"], n),
        "country":       rng.choice([f"C{i:02d}" for i in range(20)], n),
        "event_ts":      pd.Timestamp("2026-10-01")
                         + pd.to_timedelta(rng.integers(0, 31 * 24 * 3600, n), unit="s"),
        "impressions":   impressions,
        "clicks":        rng.binomial(impressions, 0.02),
        "spend":         rng.gamma(2.0, 25.0, n).round(2),
        "bid":           rng.uniform(0.1, 5.0, n).round(3),
    })


def make_advertisers(rng: np.random.Generator) -> pd.DataFrame:
    verticals = ["home", "fashion", "beauty", "food", "travel", "retail", "auto", "finance"]
    return pd.DataFrame({
        "advertiser_id": np.arange(N_ADVERTISERS),
        "vertical":      rng.choice(verticals, N_ADVERTISERS),
    })


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)

    events = make_events(N_ROWS, rng)
    events.to_csv(DATA_DIR / "ads.csv", index=False)
    # 100k-row row groups so parallel readers (Dask, DuckDB, Polars) can split the file
    events.to_parquet(DATA_DIR / "ads.parquet", index=False, row_group_size=100_000)

    make_advertisers(rng).to_parquet(DATA_DIR / "advertisers.parquet", index=False)

    for f in sorted(DATA_DIR.iterdir()):
        print(f"{f.name:22s} {f.stat().st_size / 1e6:7.1f} MB")
    print(events.dtypes)


if __name__ == "__main__":
    main()
