# prediction-markets-microstructure

Replication and inference audit of maker–taker returns in binary prediction
markets, using ~72M Kalshi trades. Reproduces the favorite–longshot bias and a
positive maker volume-weighted return, but shows that the claim **maker profit
significantly exceeds bid–ask spread compensation** is far weaker than reported
once standard errors are corrected for autocorrelation and clustering.

## Key findings

| Result | This repo | Original paper |
| --- | --- | --- |
| Favorite–longshot bias (δ<0 longshots, δ>0 favorites) | replicated, p≈0 | ✓ |
| Aggregate maker VWAR | +1.0 pp | +0.83 pp |
| Sports share of volume | 77% | 74% |
| Maker − half-spread (net alpha), volume-weighted | +0.23 pp | +0.13 pp |
| Significance of net alpha (30d rolling) | **t≈2.6 (HAC)** | t = 84.6 |

The headline `t = 84.6` comes from a 30-day **rolling** VWAR whose overlapping
daily points are heavily autocorrelated; treating them as independent inflates
the t-statistic. A Newey–West (HAC) standard error collapses it ~5×. The result
is also weighting-dependent: positive volume-weighted, negative equal-weighted.

## Data

Trades and market metadata are streamed from the public
[`TrevorJS/kalshi-trades`](https://huggingface.co/datasets/TrevorJS/kalshi-trades)
HuggingFace dataset. The first run builds a slim local parquet cache
(`kalshi_trades.parquet`, ~800 MB, git-ignored); later runs reuse it. A
HuggingFace read token (via `~/.hf_token` or the `HF_TOKEN` env var) avoids rate
limits.

## Usage

```bash
pip install -r requirements.txt

python main.py --limit 2000000   # quick smoke test
python main.py                   # full dataset (~72M trades)
python main.py --overwrite       # rebuild the cache
```

Optional date window via `analysis_config.json` (`start_date` / `end_date`).
Outputs are written to `results/tables/` (CSVs) and `results/plots/` (PNGs),
both committed for convenience.

## Method

The library (`kalshi_analysis.py`) computes **sufficient statistics in SQL**
(DuckDB) and returns small DataFrames, so every analysis scales to the full
dataset without loading raw trades into memory.

- Implied probability `p = yes_price / 100`; outcome `y = 1[result == 'yes']`.
- Calibration `δ_b = f(b) − p̄` (taker-YES) — favorite–longshot bias.
- Per-contract taker PnL: `y − p` (YES taker) or `p − y` (NO taker); maker = −taker.
- VWAR weighted by contract count; significance via monthly (clustered) and
  Newey–West HAC tests, reported both equal- and volume-weighted.

## Layout

```
kalshi_analysis.py    # library: cache build, SQL aggregations, tests, plots
main.py               # CLI entry point -> results/
analysis_config.json  # optional date-window config
sandbox.ipynb         # exploratory notebook
results/tables/       # CSV outputs
results/plots/        # figure outputs
```

## Notes / limitations

Returns are gross of fees. Results depend on the public dataset's coverage
(differs from the original paper's API pull) and on the half-spread proxy
(½·mean|Δprice|). Sports markets launched on Kalshi in late 2024, so
cross-category comparisons mix time periods.
