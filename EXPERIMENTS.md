# Experiments

`POST /api/experiments/run` executes **headless** episodes:

```
for policy in policies:
    for seed in seeds:
        run WildfireEnv to termination (or max_steps)
```

Each trial records the metric payload from `compute_public_metrics`.

## Statistics

Across seeds, for each metric: n, mean, sample SD, median, min, max, Student-t 95% CI (`scipy.stats`).

The report prints `mean ± SD`. **No hypothesis test is performed.** The API sets `significance_tested: false`. Do not claim that MARL “significantly outperforms” a baseline from this table alone.

## Fair comparison

Trials share the same scenario fields except `policy.name`, `seed`, and optional `checkpoint_path` for `marl`.

If no checkpoint is loaded, the `marl` column is an **untrained random actor** and must be read that way.

## Interactive UI default

The compare panel uses 3 seeds and 180 s so a U-series laptop can finish during a demonstration. For a paper, raise seeds (e.g. 30) and duration to the scenario of record, then export CSV.
