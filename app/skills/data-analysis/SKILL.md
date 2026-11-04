---
name: data-analysis
description: Group/aggregate, top-N, time-series trends and growth, pivot tables, correlations and outliers with pandas.
---

# Data Analysis

## Strict rules
1. **Never `read_file` a `.csv`, `.tsv` or `.xlsx` to analyze it** - it truncates large files and
   leads to wrong math. Inspect schemas, count rows and aggregate with a Python script via `execute`.
2. **Never estimate, round or mentally calculate KPIs.** Every number, percentage, anomaly and KPI
   in the final answer must first be computed and printed by a Python script.

Put every calculation in one script and print small, rounded tables with `=== headers ===`.

```python
import numpy as np

# Group + share (top 10 only)
g = (df.groupby("category", observed=True)
       .agg(total=("sales", "sum"), avg=("sales", "mean"), n=("sales", "size"))
       .sort_values("total", ascending=False))
g["share_pct"] = (g["total"] / g["total"].sum() * 100).round(1)
print("=== by category ===\n", g.head(10).round(2).to_string())

# Monthly trend (date must be datetime): print EVERY month and EVERY metric, then the
# extreme of each metric via idxmax/idxmin (lowest sales month != lowest profit month)
monthly_df = (df.groupby(df["date"].dt.to_period("M"))
                .agg(sales=("sales", "sum"), profit=("profit", "sum")).round(2))
monthly_df["sales_mom_pct"] = (monthly_df["sales"].pct_change() * 100).round(1)
print("=== monthly ===\n", monthly_df.to_string())
for col in ["sales", "profit"]:
    s = monthly_df[col]
    print(f"{col}: highest {s.idxmax()} = {s.max()}, lowest {s.idxmin()} = {s.min()}")

# Pivot with row AND column totals, plus each column's share of the grand total
pt = df.pivot_table(index="region", columns="channel", values="sales", aggfunc="sum",
                    fill_value=0, observed=True, margins=True, margins_name="Total").round(2)
print("=== region x channel ===\n", pt.to_string())
print("=== channel share % ===\n", (pt.loc["Total"] / pt.loc["Total", "Total"] * 100).round(2).to_string())

# Strong correlations, each pair once
c = df.select_dtypes("number").corr()
u = c.where(np.triu(np.ones(c.shape), k=1).astype(bool)).stack()
print("=== |r| > 0.5 ===\n", u[u.abs() > 0.5].round(3).to_string())

# IQR outliers: report count, not rows
q1, q3 = df["value"].quantile([0.25, 0.75])
out = df[(df["value"] < q1 - 1.5 * (q3 - q1)) | (df["value"] > q3 + 1.5 * (q3 - q1))]
print("outliers:", len(out))
```

Rules: vectorized pandas only (no `iterrows`), `observed=True` when grouping categories,
every number you report must come from printed output, correlation is not causation.
