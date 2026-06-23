"""Run the full Kalshi analysis and save all figures.

Examples:
    python main.py --limit 2000000          # quick smoke test
    python main.py                          # full dataset (~72M trades)
    python main.py --outdir figures --thresh 1.0
    python main.py --config analysis_config.json

The first full build streams the dataset from HuggingFace into a local parquet
cache (kalshi_trades.parquet); later runs reuse it. Provide a HF read token via
~/.hf_token or the HF_TOKEN env var to avoid rate limits.
"""
import argparse

import matplotlib
matplotlib.use('Agg')   # headless: save figures without a display

import kalshi_analysis as ka


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--limit', type=int, default=None,
                    help='rows for the cache build (default: all ~72M)')
    ap.add_argument('--cache', default='kalshi_trades.parquet', help='local cache parquet path')
    ap.add_argument('--outdir', default='results',
                    help='output dir; CSVs -> <outdir>/tables, figures -> <outdir>/plots')
    ap.add_argument('--threads', type=int, default=2, help='DuckDB threads (low avoids HTTP 429)')
    ap.add_argument('--thresh', type=float, default=1.0,
                    help='emphasize bars with |effect| >= thresh (pp)')
    ap.add_argument('--config', default='analysis_config.json',
                    help='JSON config with optional start_date and end_date filters')
    ap.add_argument('--overwrite', action='store_true', help='rebuild the cache')
    args = ap.parse_args()

    con = ka.connect()
    print('HF token:', 'loaded' if ka.load_hf_token() else 'NOT found (may hit rate limits)')
    cache = ka.build_cache(con, dst=args.cache, limit=args.limit,
                           overwrite=args.overwrite, threads=args.threads)
    cfg = ka.load_config(args.config)
    source = ka.filtered_source(f"'{cache}'", start_date=cfg['start_date'], end_date=cfg['end_date'])
    print(f"config: {args.config}")
    print(f"date filter: start_date={cfg['start_date'] or 'none'}, end_date={cfg['end_date'] or 'none'}")
    if source.startswith('(SELECT'):
        print(f"analysis window: {cfg['start_date'] or 'beginning'} to {cfg['end_date'] or 'end'}")
    else:
        print('analysis window: full cache')
    tables = ka.run_all(con, source, outdir=args.outdir, thresh=args.thresh)
    print(f"\ndone — results written to {args.outdir}/ (tables/ and plots/)")


if __name__ == '__main__':
    main()
