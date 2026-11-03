---
name: data-reporting
description: Write /output/report.md summarizing findings with tables and embedded charts. Use only when a report is requested.
---

# Data Reporting

Write `/output/report.md` with `write_file` in one step, using numbers already printed.
Do not re-run the analysis. Embed only charts that exist in /output/, with relative links.

```markdown
# <Title>

## Summary
- <3-5 bullets, each with an exact number: "Electronics drove 42.1% ($1.2M) of revenue">

## Data
- `<file>`: <rows> rows x <cols> columns, <date range>. Cleaning: <what changed, row counts>.

## Findings
### 1. <Headline>
<1-2 sentences with numbers.>

| Product | Profit |
| :--- | ---: |
| ... | ... |

![<caption>](<chart.png>)

## Caveats
- <missing data, outliers, sample size>

## Next steps
- <2-3 follow-ups>
```

Rules: lead with the insight, not the method; never invent numbers; tables of at most
15 rows; round sensibly ($1.24M, 34.5%).
