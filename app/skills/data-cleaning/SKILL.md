---
name: data-cleaning
description: Fix wrong dtypes, numbers or dates stored as text, missing values, duplicates, messy labels or column names in a pandas DataFrame.
---

# Data Cleaning

numpy and pandas are normally pre-installed. Before the first script, install them if missing
(one `execute` call; it does nothing when they are already there):

```bash
python -c "import numpy, pandas" 2>/dev/null || pip install -q numpy pandas
```

Do all fixes in the same script as the analysis. Print one audit dict, not the data.

```python
audit = {"rows_in": len(df)}
df.columns = df.columns.str.strip().str.lower().str.replace(r"\W+", "_", regex=True)

num = pd.to_numeric(df["price"].str.replace(r"[$,%]", "", regex=True), errors="coerce")
audit["price_unparsed"] = int((num.isna() & df["price"].notna()).sum())
df["price"] = num

# format="mixed" reads each date on its own. Without it pandas guesses ONE format from the
# first value, and dates written differently (2026/02/01) silently become NaT.
date = pd.to_datetime(df["date"], format="mixed", errors="coerce")
audit["date_unparsed"] = int((date.isna() & df["date"].notna()).sum())
df["date"] = date

df["region"] = df["region"].str.strip().str.title()
df.loc[df["age"].isin([-1, 999]), "age"] = pd.NA
df = df.drop_duplicates()
audit["rows_out"] = len(df)
print(audit)
```

Missing values: drop if few rows, median for skewed numbers, mode for categories,
`ffill` for sorted time series, or keep NaN when missingness matters. Record the choice.

Rules:
- Never overwrite files in /data/. Save cleaned data as `/output/cleaned.csv` if needed.
- Use `df.loc[mask, "col"] = v`, never chained indexing.
- Negative quantities are usually returns: keep them in totals unless told otherwise, and say so.
- Do not call `.astype(str)` before `.str.replace()` on columns with nulls ("nan" strings).