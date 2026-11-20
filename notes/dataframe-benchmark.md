# DataFrame Library Benchmark

pandas vs. Polars vs. PyArrow vs. DuckDB vs. Dask on the same 1M-row dataset.

## Files

| File | What it does |
|---|---|
| [`generate_data.py`](code/dataframe-benchmark/generate_data.py) | Creates `data/ads.csv` (67 MB) and `data/ads.parquet` (27 MB): 1M rows × 10 columns of dummy ads data, plus `data/advertisers.parquet` (5k rows) for the join |
| [`bench.py`](code/dataframe-benchmark/bench.py) | Times 7 tasks in every library and writes `results.md` |
| `results.md` | The latest results table (see below) |
| [`requirements.txt`](code/dataframe-benchmark/requirements.txt) | Libraries to install |

**Columns:** `user_id, ad_id, advertiser_id, surface, country, event_ts, impressions, clicks, spend, bid`

## Run it

```bash
cd Concepts/dataframe-benchmark
.venv/bin/python generate_data.py   # only needed once
.venv/bin/python bench.py
```

The `.venv` uses Python 3.13. To rebuild it, use any Python ≥ 3.11 (pandas 3.x requires it):

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Results

**Setup:**
- Apple M5 Pro, 18 cores, 48 GB RAM.
- Python 3.13.18 · pandas 3.1.1 · Polars 2.1.0 · PyArrow 26.0.1 · Dask 2026.8.0 · DuckDB 2.0.0.
- Median of 5 runs after a warm-up, in **milliseconds** (lower is better). Fastest per task in **bold**.
- Every library returns identical answers: 102,032 filtered rows, 5,000 groups, 8 verticals.

| Library | read_csv | read_parquet | filter | groupby | sort_top100 | join_groupby | write_parquet |
|---|---:|---:|---:|---:|---:|---:|---:|
| pandas (NumPy dtypes) | 18 | 7.3 | 3.0 | 8.1 | 82 | 21 | 131 |
| pandas (PyArrow dtypes) | 15 | 5.4 | 2.3 | 8.8 | 135 | 26 | 127 |
| Polars (eager) | 16 | 3.5 | **1.1** | 7.1 | 16 | 6.0 | **17** |
| Polars (lazy, from file) | 16 | **3.5** | 1.5 | 4.1 | 5.7 | 4.2 | 19 |
| PyArrow | **13** | 5.0 | 3.9 | **1.4** | 124 | 14 | 129 |
| DuckDB (in-memory Arrow) | 92 | 7.7 | 2.0 | 2.8 | **2.4** | **2.5** | 28 |
| DuckDB (from file) | – | – | 2.3 | 4.2 | 5.1 | 3.5 | – |
| Dask (from file) | 54 | 39 | 20 | 23 | 45 | 46 | 187 |

**Tasks:**
- `filter`: count rows where surface = search and spend > 50.
- `groupby`: per advertiser, sum impressions and clicks, and average spend.
- `sort_top100`: sort by spend, keep the top 100 rows.
- `join_groupby`: join to advertisers, then total spend per vertical.
- **"From file"** rows include reading the Parquet file; lazy engines are normally used that way. The other rows run on data already in memory.

## What I learned

1. **At 1M rows, every library is fast** (milliseconds). At this size, choose by ease of use and team familiarity, not speed. The differences start to matter at 10M–1B rows.
2. **Polars is the best all-rounder.** It's fastest or close on almost every task, and writes Parquet about **8× faster** than pandas.
3. **Lazy beats eager** (Polars lazy vs. eager on groupby and sort). The query optimizer reads **only the columns the query needs** from Parquet ("projection pushdown") and plans the whole query at once. That's why *reading from the file* can beat data that's *already in memory*.
4. **DuckDB wins on sort and join.** `ORDER BY … LIMIT 100` becomes a **top-K** operation instead of a full sort. You write it in SQL, and it can query Parquet files directly.
5. **The "sort" column isn't a fair fight.** pandas, PyArrow, and Polars eager do a **full sort**, while DuckDB and Polars lazy only find the top 100. In pandas, `df.nlargest(100, "spend")` is the top-K equivalent.
6. **PyArrow is mainly a format and I/O layer.** Its CSV read and groupby are fast, but its API is low-level and sorting is slow. pandas writes Parquet *through* PyArrow, which is why their write times are the same.
7. **PyArrow dtypes in pandas aren't automatically faster.** They gave a faster CSV read and filter here, but slower sort and groupby. Measure before switching.
8. **Dask is slowest at this size.** Scheduling its tasks has overhead. Dask is built for data **larger than memory** or for **clusters**.

## Same task, every library: groupby + aggregation

```python
# pandas
df.groupby("advertiser_id").agg(impressions=("impressions", "sum"),
                                clicks=("clicks", "sum"),
                                spend=("spend", "mean"))

# Polars (eager); for lazy, start from pl.scan_parquet(...) and end with .collect()
df.group_by("advertiser_id").agg(pl.col("impressions").sum(),
                                 pl.col("clicks").sum(),
                                 pl.col("spend").mean())

# PyArrow
tbl.group_by("advertiser_id").aggregate([("impressions", "sum"),
                                         ("clicks", "sum"),
                                         ("spend", "mean")])

# DuckDB (SQL, directly on a Parquet file, a pandas/Polars DataFrame, or an Arrow table)
duckdb.sql("""SELECT advertiser_id, sum(impressions), sum(clicks), avg(spend)
              FROM 'ads.parquet' GROUP BY advertiser_id""")

# Dask (pandas-like API; nothing runs until .compute())
dd.read_parquet("ads.parquet").groupby("advertiser_id").agg(
    {"impressions": "sum", "clicks": "sum", "spend": "mean"}).compute()
```

## When to use which

| Situation | Use |
|---|---|
| Interviews; small or medium data; the most libraries and examples | **pandas** |
| Fast single-machine pipelines on 10M–1B rows | **Polars** (prefer the lazy API) |
| SQL-style analysis on Parquet or CSV files, without loading them first | **DuckDB** |
| Reading and writing formats; passing data between tools without copying | **PyArrow** |
| Data larger than memory, or a cluster, while keeping a pandas-like API | **Dask** (or **PySpark**) |

## Results

Python 3.13.18 · pandas 3.1.1 · polars 2.1.0 · pyarrow 26.0.1 · dask 2026.8.0 · duckdb 2.0.0

Median of 5 runs, in milliseconds (lower is better). Fastest per task in **bold**.

| Library | read_csv | read_parquet | filter | groupby | sort_top100 | join_groupby | write_parquet |
|---|---:|---:|---:|---:|---:|---:|---:|
| pandas (NumPy dtypes) | 18 | 7.3 | 3.0 | 8.1 | 82 | 21 | 131 |
| pandas (PyArrow dtypes) | 15 | 5.4 | 2.3 | 8.8 | 135 | 26 | 127 |
| Polars (eager) | 16 | 3.5 | **1.1** | 7.1 | 16 | 6.0 | **17** |
| Polars (lazy, from file) | 16 | **3.5** | 1.5 | 4.1 | 5.7 | 4.2 | 19 |
| PyArrow | **13** | 5.0 | 3.9 | **1.4** | 124 | 14 | 129 |
| DuckDB (in-memory Arrow) | 92 | 7.7 | 2.0 | 2.8 | **2.4** | **2.5** | 28 |
| DuckDB (from file) | – | – | 2.3 | 4.2 | 5.1 | 3.5 | – |
| Dask (from file) | 54 | 39 | 20 | 23 | 45 | 46 | 187 |
