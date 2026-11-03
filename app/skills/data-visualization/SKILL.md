---
name: data-visualization
description: Draw bar, line, histogram, box, scatter or heatmap charts with matplotlib/seaborn in one script and save them as PNG.
---

# Data Visualization

matplotlib and seaborn are normally pre-installed. Before the first script, install them if missing
(one `execute` call; it does nothing when they are already there):

```bash
python -c "import matplotlib, seaborn" 2>/dev/null || pip install -q matplotlib seaborn
```

One script for all charts. Print only the saved paths.

Plot ONLY the prepared CSV you were given - the main agent already cleaned it: an aggregated
table for bar and line charts, or the cleaned rows (/output/cleaned.csv) for histograms, box
plots and scatter plots. Never read /data/ and never clean, filter or re-aggregate anything:
the chart must show exactly the numbers in the answer.

```python
import matplotlib
matplotlib.use("Agg")  # before pyplot
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import pandas as pd
import seaborn as sns
from pathlib import Path

OUT = Path("/output"); OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

def save(fig, name):
    fig.savefig(OUT / name, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("saved", OUT / name)

# Grouped bars: compare several measures per category
# The prepared table: first column = category, other columns = values. Plotted as it is.
agg = pd.read_csv("/output/revenue_profit_by_product_data.csv", index_col=0)
agg = agg.sort_values(agg.columns[-1], ascending=False)  # sorting only - never drop or add rows
fig, ax = plt.subplots(figsize=(9, 5))
agg.plot.bar(ax=ax)
ax.set_title("Net revenue and profit by product", weight="bold"); ax.set_xlabel("")
ax.yaxis.set_major_formatter(mt.StrMethodFormatter("{x:,.0f}"))
save(fig, "revenue_profit_by_product.png")
```

Keep category labels exactly as written in the prepared CSV.

Chart choice: categories -> sorted bar (the prepared CSV already has at most 10 rows + "Other"); over time -> line
(convert PeriodIndex with `.to_timestamp()`); one numeric -> `sns.histplot`; groups ->
boxplot (rows already limited to the top 10 groups); two numerics -> `sns.regplot(ci=None)`, sample 5000 rows if large;
correlations -> `sns.heatmap(annot=True, fmt=".2f", center=0)`; avoid pie charts.

Always: title, axis labels with units, thousands separators, bar axes from zero.