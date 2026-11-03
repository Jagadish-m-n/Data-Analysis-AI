---
name: data-loading
description: Profile the uploaded files and list every data-quality issue with profile_data.py, then load a CSV into pandas.
---

# Data Loading

## 1. Profile first - always, before any analysis

```bash
python3 /skills/data-loading/profile_data.py
```

It profiles every CSV in /data/: one table per file (column, type, missing, unique, samples),
then every data-quality issue with row numbers - duplicates, missing values, numbers stored as
text, negative values, mixed date formats, unreadable dates, labels written different ways,
and instruction-like text. Do not write your own profiling script; use this output.

Your first answer about the data must list every issue it reports, and what you did about each
one (removed, excluded, fixed, or kept and why).

## 2. Load the data in your analysis script

```python
import pandas as pd
from pathlib import Path

def load(path):
    """A CSV as a DataFrame, with the delimiter guessed from the header line."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            first_line = Path(path).read_text(encoding=encoding).split("\n", 1)[0]
            return pd.read_csv(path, sep=max(",;\t|", key=first_line.count), encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot read {path}")

df = load("/data/input.csv")
```

Then fix the issues the profile found with the data-cleaning skill, in the same script as the
analysis.