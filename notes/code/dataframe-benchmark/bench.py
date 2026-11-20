"""Benchmark pandas, Polars, PyArrow, Dask, and DuckDB on the same 1M-row ads dataset.

Each task is run REPEATS times after one warm-up run; the median wall time is reported.
Every task fully materializes its result (no lazy results left unevaluated).

Tasks
  read_csv      load ads.csv into memory
  read_parquet  load ads.parquet into memory
  filter        count rows where surface == 'search' and spend > 50
  groupby       per advertiser: sum impressions, sum clicks, mean spend
  sort_top100   sort by spend descending, take top 100 rows
  join_groupby  join to advertisers on advertiser_id, then total spend per vertical
  write_parquet write the full table to Parquet

Filter / groupby / sort / join run on data already in memory, except for the lazy
engines (Polars lazy, Dask, DuckDB-on-file), which are timed end to end from the
Parquet file because reading is part of how they're used.
"""
import gc
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import dask
import dask.dataframe as dd
import duckdb
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

DATA = Path(__file__).parent / "data"
CSV, PARQUET, ADV = str(DATA / "ads.csv"), str(DATA / "ads.parquet"), str(DATA / "advertisers.parquet")
REPEATS = 5
TMP = Path(tempfile.mkdtemp())


def timeit(fn):
    fn()  # warm-up
    times = []
    for _ in range(REPEATS):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def duck_to_arrow(rel):
    """DuckDB relation -> Arrow table (method name differs across versions)."""
    for name in ("to_arrow_table", "fetch_arrow_table", "arrow"):
        if hasattr(rel, name):
            out = getattr(rel, name)()
            return out.read_all() if hasattr(out, "read_all") else out
    raise AttributeError("no arrow export on DuckDB relation")


# ---------------------------------------------------------------- in-memory data
pdf = pd.read_parquet(PARQUET)
pdf_arrow = pd.read_parquet(PARQUET, dtype_backend="pyarrow")
adv_pd = pd.read_parquet(ADV)
adv_pd_arrow = pd.read_parquet(ADV, dtype_backend="pyarrow")
plf = pl.read_parquet(PARQUET)
adv_pl = pl.read_parquet(ADV)
tbl = pq.read_table(PARQUET)
adv_tbl = pq.read_table(ADV)
con = duckdb.connect()
con.register("ads", tbl)
con.register("advertisers", adv_tbl)

# ---------------------------------------------------------------- task definitions
SQL = {
    "filter": "SELECT count(*) FROM {t} WHERE surface = 'search' AND spend > 50",
    "groupby": """SELECT advertiser_id, sum(impressions) AS impressions,
                         sum(clicks) AS clicks, avg(spend) AS spend
                  FROM {t} GROUP BY advertiser_id""",
    "sort_top100": "SELECT * FROM {t} ORDER BY spend DESC LIMIT 100",
    "join_groupby": """SELECT a.vertical, sum(e.spend) AS spend
                       FROM {t} e JOIN {adv} a USING (advertiser_id)
                       GROUP BY a.vertical""",
}


def pandas_tasks(df, adv, engine_kwargs):
    return {
        "read_csv":      lambda: pd.read_csv(CSV, **engine_kwargs.get("csv", {})),
        "read_parquet":  lambda: pd.read_parquet(PARQUET, **engine_kwargs.get("parquet", {})),
        "filter":        lambda: int(((df["surface"] == "search") & (df["spend"] > 50)).sum()),
        "groupby":       lambda: df.groupby("advertiser_id").agg(
                             impressions=("impressions", "sum"),
                             clicks=("clicks", "sum"),
                             spend=("spend", "mean")),
        "sort_top100":   lambda: df.sort_values("spend", ascending=False).head(100),
        "join_groupby":  lambda: df.merge(adv, on="advertiser_id").groupby("vertical")["spend"].sum(),
        "write_parquet": lambda: df.to_parquet(TMP / "pd.parquet", index=False),
    }


LIBS = {
    "pandas (NumPy dtypes)": pandas_tasks(pdf, adv_pd, {}),
    "pandas (PyArrow dtypes)": pandas_tasks(
        pdf_arrow, adv_pd_arrow,
        {"csv": {"engine": "pyarrow", "dtype_backend": "pyarrow"},
         "parquet": {"dtype_backend": "pyarrow"}}),
    "Polars (eager)": {
        "read_csv":      lambda: pl.read_csv(CSV),
        "read_parquet":  lambda: pl.read_parquet(PARQUET),
        "filter":        lambda: plf.filter((pl.col("surface") == "search") & (pl.col("spend") > 50)).height,
        "groupby":       lambda: plf.group_by("advertiser_id").agg(
                             pl.col("impressions").sum(), pl.col("clicks").sum(), pl.col("spend").mean()),
        "sort_top100":   lambda: plf.sort("spend", descending=True).head(100),
        "join_groupby":  lambda: plf.join(adv_pl, on="advertiser_id").group_by("vertical").agg(pl.col("spend").sum()),
        "write_parquet": lambda: plf.write_parquet(TMP / "pl.parquet"),
    },
    "Polars (lazy, from file)": {
        "read_csv":      lambda: pl.scan_csv(CSV).collect(),
        "read_parquet":  lambda: pl.scan_parquet(PARQUET).collect(),
        "filter":        lambda: pl.scan_parquet(PARQUET)
                             .filter((pl.col("surface") == "search") & (pl.col("spend") > 50))
                             .select(pl.len()).collect().item(),
        "groupby":       lambda: pl.scan_parquet(PARQUET).group_by("advertiser_id").agg(
                             pl.col("impressions").sum(), pl.col("clicks").sum(), pl.col("spend").mean()).collect(),
        "sort_top100":   lambda: pl.scan_parquet(PARQUET).sort("spend", descending=True).head(100).collect(),
        "join_groupby":  lambda: pl.scan_parquet(PARQUET).join(pl.scan_parquet(ADV), on="advertiser_id")
                             .group_by("vertical").agg(pl.col("spend").sum()).collect(),
        "write_parquet": lambda: pl.scan_parquet(PARQUET).sink_parquet(TMP / "pl_lazy.parquet"),
    },
    "PyArrow": {
        "read_csv":      lambda: pacsv.read_csv(CSV),
        "read_parquet":  lambda: pq.read_table(PARQUET),
        "filter":        lambda: tbl.filter(pc.and_(pc.equal(tbl["surface"], "search"),
                                                    pc.greater(tbl["spend"], 50))).num_rows,
        "groupby":       lambda: tbl.group_by("advertiser_id").aggregate(
                             [("impressions", "sum"), ("clicks", "sum"), ("spend", "mean")]),
        "sort_top100":   lambda: tbl.sort_by([("spend", "descending")]).slice(0, 100),
        "join_groupby":  lambda: tbl.join(adv_tbl, "advertiser_id").group_by("vertical").aggregate([("spend", "sum")]),
        "write_parquet": lambda: pq.write_table(tbl, TMP / "pa.parquet"),
    },
    "DuckDB (in-memory Arrow)": {
        "read_csv":      lambda: duck_to_arrow(con.read_csv(CSV)),
        "read_parquet":  lambda: duck_to_arrow(con.read_parquet(PARQUET)),
        **{k: (lambda q=q: duck_to_arrow(con.sql(q.format(t="ads", adv="advertisers")))) for k, q in SQL.items()},
        "write_parquet": lambda: con.sql(f"COPY ads TO '{TMP / 'duck.parquet'}' (FORMAT parquet)"),
    },
    "DuckDB (from file)": {
        **{k: (lambda q=q: duck_to_arrow(con.sql(q.format(t=f"'{PARQUET}'", adv=f"'{ADV}'")))) for k, q in SQL.items()},
    },
    "Dask (from file)": {
        "read_csv":      lambda: dd.read_csv(CSV).compute(),
        "read_parquet":  lambda: dd.read_parquet(PARQUET).compute(),
        "filter":        lambda: (lambda d: int(((d["surface"] == "search") & (d["spend"] > 50)).sum().compute()))(
                             dd.read_parquet(PARQUET)),
        "groupby":       lambda: dd.read_parquet(PARQUET).groupby("advertiser_id").agg(
                             {"impressions": "sum", "clicks": "sum", "spend": "mean"}).compute(),
        "sort_top100":   lambda: dd.read_parquet(PARQUET).nlargest(100, "spend").compute(),
        "join_groupby":  lambda: dd.read_parquet(PARQUET).merge(dd.read_parquet(ADV), on="advertiser_id")
                             .groupby("vertical")["spend"].sum().compute(),
        "write_parquet": lambda: dd.read_parquet(PARQUET).to_parquet(TMP / "dask_out", write_index=False),
    },
}

TASKS = ["read_csv", "read_parquet", "filter", "groupby", "sort_top100", "join_groupby", "write_parquet"]


def main():
    versions = {m.__name__: m.__version__ for m in (pd, pl, pa, dask, duckdb)}
    print(f"Python {sys.version.split()[0]} | " + " | ".join(f"{k} {v}" for k, v in versions.items()))
    print(f"Rows: {len(pdf):,}  Columns: {pdf.shape[1]}  Repeats: {REPEATS} (median)\n")

    results = {}
    for lib, tasks in LIBS.items():
        results[lib] = {}
        for task in TASKS:
            if task not in tasks:
                results[lib][task] = None
                continue
            try:
                results[lib][task] = timeit(tasks[task])
            except Exception as e:  # record failures instead of aborting the whole run
                results[lib][task] = f"ERR: {type(e).__name__}: {e}"[:80]
            r = results[lib][task]
            print(f"{lib:26s} {task:14s} {r if isinstance(r, str) else f'{r * 1000:9.1f} ms'}")

    # markdown table, ms, fastest per task in bold
    lines = ["| Library | " + " | ".join(TASKS) + " |", "|---|" + "---:|" * len(TASKS)]
    best = {t: min((r[t] for r in results.values() if isinstance(r[t], float)), default=None) for t in TASKS}
    for lib, r in results.items():
        cells = []
        for t in TASKS:
            v = r[t]
            if v is None:
                cells.append("–")
            elif isinstance(v, str):
                cells.append("error")
            else:
                s = f"{v * 1000:,.0f}" if v >= 0.01 else f"{v * 1000:.1f}"
                cells.append(f"**{s}**" if v == best[t] else s)
        lines.append(f"| {lib} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    header = (f"Python {sys.version.split()[0]} · " + " · ".join(f"{k} {v}" for k, v in versions.items())
              + f"\n\nMedian of {REPEATS} runs, in milliseconds (lower is better). Fastest per task in **bold**.\n\n")
    (Path(__file__).parent / "results.md").write_text(header + table + "\n")
    print("\n" + table)
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
