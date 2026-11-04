---
name: data-reporting
description: Write /output/report.md summarizing findings with tables and embedded charts. Use only when a report is requested.
---

# Data Reporting

Write `/output/report.md` with `write_file` in one step, copying every number exactly as a
script printed it. Do not re-run the whole analysis: if a number you need was never printed, or
is no longer in an output you can see, first run one short script that prints it. Embed only
charts that `ls /output/` shows, with relative links.

```markdown
# <Title>

## Summary
- <3-5 bullets, each with an exact printed number: "Electronics drove 42.15% (1,236,543.21) of revenue">

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

Rules: lead with the insight, not the method; never invent, estimate or round numbers yourself
(round in the script, then copy what it printed); never add a currency symbol the data does not
use; tables of at most 15 rows.
