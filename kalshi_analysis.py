"""Kalshi prediction-market analysis library (DuckDB-backed).

Reproduces the calibration / maker-taker / VWAR analyses from the paper, scaled to
the full dataset by computing sufficient statistics in SQL and returning small
DataFrames. Plotting functions save figures to disk. See main.py for a runnable
entry point.

Conventions:
- Implied probability p = yes_price / 100; outcome y = 1[result == 'yes'].
- Calibration delta_b = f(b) - p_bar is taker-YES (paper's FLSB definition).
- VWAR / maker-taker uses ALL trades with signed per-contract PnL:
    taker buys YES at p   -> y - p
    taker buys NO at (1-p) -> p - y
  maker PnL = -taker PnL. Volume weight = `count` (contracts per trade).
"""
import os
import re
import pathlib
import json
import datetime as dt

import numpy as np
import pandas as pd
from scipy import stats
import duckdb
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Data sources & bucketing constants
# ---------------------------------------------------------------------------
TRADES_GLOB  = 'hf://datasets/TrevorJS/kalshi-trades/trades-*.parquet'
MARKETS_GLOB = 'hf://datasets/TrevorJS/kalshi-trades/markets-*.parquet'
DEFAULT_CONFIG = {
    'start_date': None,
    'end_date': None,
}

PRICE_BIN = 5   # cent width of price buckets
TTC_CASE = ("CASE WHEN ttc_h < 6 THEN '0-6h' WHEN ttc_h < 24 THEN '6h-1d' "
            "WHEN ttc_h < 72 THEN '1-3d' WHEN ttc_h < 168 THEN '3-7d' "
            "WHEN ttc_h < 720 THEN '7-30d' ELSE '30d+' END")
TTC_ORDER = ['0-6h', '6h-1d', '1-3d', '3-7d', '7-30d', '30d+']
LONGSHOT, FAVORITE = 0.25, 0.75

# ---------------------------------------------------------------------------
# Ticker -> broad category mapping (paper-style)
# ---------------------------------------------------------------------------
_P = r'^(?:KX)?'   # optional Kalshi 'KX' prefix
SPORTS_LEAGUES = (r'(NFL|NBA|WNBA|NHL|MLB|NCAA|CFB|CBB|CBA|NBL|UFC|BOXING|MMA|SOCCER|'
                  r'EPL|UCL|UEL|MLS|FIFA|EUROLEAGUE|SAUDIPL|AFCON|SERIEA|LALIGA|'
                  r'BUNDESLIGA|LIGUE1|COPA|CONCACAF|IPL|T20|CRICKET|RUGBY|TENNIS|ATP|'
                  r'WTA|GOLF|PGA|LPGA|F1|MOTOGP|NASCAR|OLYMPIC|MVE)')
SPORTS_DESC = r'(SINGLEGAME|MULTIGAME|GAME|COACH|SPREAD|MATCHUP|PLAYOFF)'
OTHER_RULES = [
    ('Crypto',        r'(BTC|ETH|SOL|XRP|DOGE|CRYPTO)'),       # ADA dropped (collides w/ ADAMS)
    ('Politics',      r'(PRES|SENATE|HOUSE|GOV|ELECT|POLL|CONGRESS|SCOTUS|IMPEACH|'
                      r'SHUTDOWN|DEBTCEIL|SLOAN|GASTAX|TRUMP|BIDEN|HARRIS|ADAMS|MAYOR|'
                      r'APPROVE|538)'),
    ('World Events',  r'(TRDDEF|TRADE|WAR|UKRAINE|RUSSIA|CHINA|OPEC|CO2|EMISS|NUKE)'),
    ('Weather',       r'(TEMP|HIGH|LOW|RAIN|SNOW|HURR|STORM|WEATHER)'),
    ('Entertainment', r'(OSCAR|EMMY|GRAMMY|MOVIE|BOXOFFICE|ROTTEN|SPOTIFY|BILLBOARD|'
                      r'TVRATING|NETFLIX|1SONG|SONG|ALBUM)'),
    ('Science/Tech',  r'(SPACE|NASA|ROCKET|GPT|IPHONE|TESLA|TECH)'),
    ('Finance',       r'(CPI|PCE|GDP|FED|FOMC|RATE|JOBS|PAYROLL|UNEMP|GBP|EUR|JPY|USD|'
                      r'FX|NASDAQ|SPX|SP500|DJIA|INX|INDU|GAS|OIL|WTI|BRENT|GOLD|HOME|'
                      r'MORTG|RECESS|10Y|2Y|30Y|YIELD|TREAS)'),
]


def categorize(ticker: str) -> str:
    """Map a Kalshi ticker (or series prefix) to a broad paper-style category."""
    s = (ticker or '').split('-')[0].upper().strip()   # series code
    if not s:
        return 'Other'
    if re.match(_P + SPORTS_LEAGUES, s) or re.search(SPORTS_DESC, s):
        return 'Sports'
    for cat, toks in OTHER_RULES:
        if re.match(_P + toks, s):
            return cat
    return 'Other'


# ---------------------------------------------------------------------------
# Connection / token / cache
# ---------------------------------------------------------------------------
def connect():
    """DuckDB connection with httpfs loaded."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    return con


def load_hf_token(path='~/.hf_token'):
    """Load a HuggingFace token into HF_TOKEN from env or a saved file. Returns bool."""
    if os.environ.get('HF_TOKEN'):
        return True
    f = pathlib.Path(path).expanduser()
    if f.exists():
        os.environ['HF_TOKEN'] = f.read_text().strip()
        return True
    return False


def load_config(path='analysis_config.json'):
    """Load analysis settings from JSON, returning defaults when the file is absent."""
    cfg = DEFAULT_CONFIG.copy()
    if not path:
        return cfg
    f = pathlib.Path(path)
    if not f.exists():
        return cfg
    with f.open() as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f'Config must be a JSON object: {path}')
    unknown = set(loaded) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f'Unknown config key(s): {", ".join(sorted(unknown))}')
    cfg.update(loaded)
    return cfg


def _date_literal(value, key):
    """Validate a YYYY-MM-DD config date and return a DuckDB timestamp literal."""
    if value in (None, ''):
        return None
    try:
        d = dt.date.fromisoformat(str(value))
    except ValueError as e:
        raise ValueError(f'{key} must use YYYY-MM-DD format, got {value!r}') from e
    return d


def filtered_source(source, start_date=None, end_date=None):
    """Wrap a source table/path with an optional inclusive date window on created_s."""
    start = _date_literal(start_date, 'start_date')
    end = _date_literal(end_date, 'end_date')
    if start and end and start > end:
        raise ValueError(f'start_date {start} cannot be after end_date {end}')

    where = []
    if start:
        where.append(f"created_s >= epoch(TIMESTAMP '{start.isoformat()} 00:00:00')")
    if end:
        after_end = end + dt.timedelta(days=1)
        where.append(f"created_s < epoch(TIMESTAMP '{after_end.isoformat()} 00:00:00')")
    if not where:
        return source
    return f"(SELECT * FROM {source} WHERE {' AND '.join(where)})"


def _epoch_expr(typestr: str, col: str) -> str:
    """SQL expression converting a timestamp/unix-int/ISO-string column to epoch seconds."""
    t = typestr.upper()
    if 'TIMESTAMP' in t or 'DATE' in t:
        return f"epoch({col})"
    if any(k in t for k in ('INT', 'DEC', 'DOUBLE', 'FLOAT', 'REAL')):
        return f"(CASE WHEN {col} > 1e12 THEN {col}/1000.0 ELSE CAST({col} AS DOUBLE) END)"
    return f"epoch(TRY_CAST({col} AS TIMESTAMP))"


def _try(con, sql):
    """Run a SET/PRAGMA, ignoring it if this DuckDB build doesn't know the option."""
    try:
        con.execute(sql)
    except Exception as e:
        print(f"  (skipped: {sql} -> {str(e)[:60]})")


def build_cache(con, dst='kalshi_trades.parquet', limit=None, overwrite=False,
                threads=2, hf_token=None):
    """Process trades x markets once into a slim local parquet (the full-dataset scan).

    Keeps only needed columns, normalizes timestamps to epoch seconds, pre-filters
    to finalized markets, and writes locally so later analysis avoids re-streaming.
    Lowers `threads` and enables retry/backoff to survive HuggingFace HTTP 429s; reads
    HF_TOKEN from the environment for a higher rate limit. limit=None builds all ~72M.
    """
    if os.path.exists(dst) and not overwrite:
        n = con.execute(f"SELECT count(*) FROM '{dst}'").fetchone()[0]
        print(f"cache exists: {dst} ({n:,} rows). Use overwrite=True to rebuild.")
        return dst

    con.execute("PRAGMA enable_progress_bar;")
    con.execute(f"SET threads TO {threads};")
    for s in ("SET http_retries=10", "SET http_retry_wait_ms=2000",
              "SET http_retry_backoff=2", "SET enable_http_metadata_cache=true",
              "SET http_keep_alive=true"):
        _try(con, s)
    hf_token = hf_token or os.environ.get('HF_TOKEN')
    if hf_token:
        _try(con, f"CREATE OR REPLACE SECRET hf (TYPE HUGGINGFACE, TOKEN '{hf_token}')")

    ct  = con.execute(f"SELECT typeof(created_time) FROM '{TRADES_GLOB}' LIMIT 1").fetchone()[0]
    clt = con.execute(f"SELECT typeof(close_time) FROM '{MARKETS_GLOB}' LIMIT 1").fetchone()[0]
    created_s, close_s = _epoch_expr(ct, 't.created_time'), _epoch_expr(clt, 'm.close_time')
    lim = f"LIMIT {int(limit)}" if limit else ""

    tmp = dst + '.tmp'                                  # write atomically: no half-built cache
    con.execute(f"""
        COPY (
            SELECT t.ticker, t.count, t.yes_price, t.taker_side,
                   m.result, split_part(t.ticker, '-', 1) AS series,
                   {created_s} AS created_s, {close_s} AS close_s
            FROM '{TRADES_GLOB}' t
            JOIN (SELECT ticker, result, close_time FROM '{MARKETS_GLOB}'
                  WHERE result IN ('yes','no')) m USING (ticker)
            WHERE t.taker_side IN ('yes','no')
            {lim}
        ) TO '{tmp}' (FORMAT parquet);
    """)
    os.replace(tmp, dst)
    n = con.execute(f"SELECT count(*) FROM '{dst}'").fetchone()[0]
    print(f"cached {n:,} rows -> {dst}")
    return dst


# ---------------------------------------------------------------------------
# SQL aggregations -> small DataFrames
# ---------------------------------------------------------------------------
def _add_ttest(df, mean='delta', sd='sd_dev', n='n'):
    """Add t-stat and two-sided p-value from per-group (mean, std, n)."""
    df = df.copy()
    df['t'] = df[mean] / (df[sd] / np.sqrt(df[n]))
    df['pval'] = 2 * stats.t.sf(np.abs(df['t']), (df[n] - 1).clip(lower=1))
    return df


def _taker_pnl(a=''):
    """Per-contract TAKER gross PnL (maker = -taker). `a` is an optional table alias.
    taker buys YES at p -> (y - p); taker buys NO at (1-p) -> (p - y)."""
    p = (a + '.') if a else ''
    return (f"(CASE WHEN {p}taker_side='yes' THEN ({p}result='yes')::INT - {p}yes_price/100.0 "
            f"ELSE {p}yes_price/100.0 - ({p}result='yes')::INT END)")


# Kalshi per-contract trading fees = rate * p * (1-p)  (p = yes_price/100; side-symmetric)
TAKER_FEE_RATE = 0.07
MAKER_FEE_RATE = 0.0175


def _fee(rate, a=''):
    """Per-contract fee expression: rate * p * (1-p), p = yes_price/100."""
    p = (a + '.') if a else ''
    return f"({rate} * ({p}yes_price/100.0) * (1 - {p}yes_price/100.0))"


def _taker_pnl_net(a=''):
    """Taker PnL net of the taker fee."""
    return f"({_taker_pnl(a)} - {_fee(TAKER_FEE_RATE, a)})"


def _maker_pnl_net(a=''):
    """Maker PnL net of the maker fee (= -taker_gross - maker_fee; NOT -taker_net)."""
    return f"(-({_taker_pnl(a)}) - {_fee(MAKER_FEE_RATE, a)})"


def calibration_table(con, source, side='yes'):
    """Per 5c price bucket: pbar, f(b), delta_b = f - pbar, with one-sample t-test."""
    q = f"""
        SELECT (yes_price // {PRICE_BIN}) * {PRICE_BIN} / 100.0 AS bin_lo,
               count(*) AS n, sum(count) AS contracts,
               avg(yes_price/100.0) AS pbar, avg((result='yes')::INT) AS f,
               avg((result='yes')::INT - yes_price/100.0) AS delta,
               stddev_samp((result='yes')::INT - yes_price/100.0) AS sd_dev
        FROM {source} WHERE taker_side = '{side}'
        GROUP BY 1 ORDER BY 1
    """
    cal = _add_ttest(con.execute(q).df())
    cal['mid'] = cal['bin_lo'] + PRICE_BIN / 200.0
    return cal


def calibration_by_ttc(con, source, side='yes'):
    """delta_b per (time-to-close bucket x price bucket), with t-test."""
    q = f"""
        WITH base AS (
            SELECT (yes_price // {PRICE_BIN}) * {PRICE_BIN} / 100.0 AS bin_lo,
                   (close_s - created_s)/3600.0 AS ttc_h,
                   (result='yes')::INT - yes_price/100.0 AS dev
            FROM {source} WHERE taker_side = '{side}' AND close_s >= created_s
        )
        SELECT {TTC_CASE} AS ttc_bucket, bin_lo, count(*) AS n,
               avg(dev) AS delta, stddev_samp(dev) AS sd_dev
        FROM base GROUP BY 1, 2 ORDER BY 2
    """
    d = _add_ttest(con.execute(q).df())
    d['mid'] = d['bin_lo'] + PRICE_BIN / 200.0
    return d


def deltatime_regression(con, source, side='yes'):
    """Per price bucket: OLS of deviation (y-p) on time-to-close (days), via SQL regr_*."""
    q = f"""
        WITH base AS (
            SELECT (yes_price // {PRICE_BIN}) * {PRICE_BIN} / 100.0 AS bin_lo,
                   (close_s - created_s)/86400.0 AS ttc_d,
                   (result='yes')::INT - yes_price/100.0 AS dev
            FROM {source} WHERE taker_side = '{side}' AND close_s >= created_s
        )
        SELECT bin_lo, regr_count(dev, ttc_d) AS n,
               regr_slope(dev, ttc_d) AS slope, regr_intercept(dev, ttc_d) AS intercept,
               regr_r2(dev, ttc_d) AS r2, regr_sxx(dev, ttc_d) AS sxx,
               regr_syy(dev, ttc_d) AS syy, regr_sxy(dev, ttc_d) AS sxy
        FROM base GROUP BY 1 ORDER BY 1
    """
    r = con.execute(q).df()
    r['mid'] = r['bin_lo'] + PRICE_BIN / 200.0
    denom = (r['n'] - 2).clip(lower=1) * r['sxx']                 # slope standard error
    r['se'] = np.sqrt(((r['syy'] - r['slope'] * r['sxy']) / denom).clip(lower=0))
    r['t'] = r['slope'] / r['se']
    r['pval'] = 2 * stats.t.sf(np.abs(r['t']), (r['n'] - 2).clip(lower=1))
    r['slope_pp_day'] = r['slope'] * 100
    r['intercept_pp'] = r['intercept'] * 100
    return r[['mid', 'bin_lo', 'n', 'slope_pp_day', 'intercept_pp', 'r2', 'se', 't', 'pval']]


def vwar_by_price(con, source):
    """Volume-weighted taker/maker return (pp) per price bucket, over ALL trades."""
    q = f"""
        SELECT (yes_price // {PRICE_BIN}) * {PRICE_BIN} / 100.0 AS bin_lo,
               sum({_taker_pnl()} * count) / sum(count) * 100 AS vwar_pp,
               sum(count) AS contracts, count(*) AS n
        FROM {source}
        GROUP BY 1 ORDER BY 1
    """
    t = con.execute(q).df()
    t['maker_vwar_pp'] = -t['vwar_pp']
    t['mid'] = t['bin_lo'] + PRICE_BIN / 200.0
    return t


def vwar_grid(con, source):
    """Volume-weighted taker return (pp) on the time-to-close x price grid, over ALL trades."""
    q = f"""
        WITH base AS (
            SELECT (yes_price // {PRICE_BIN}) * {PRICE_BIN} / 100.0 AS bin_lo,
                   (close_s - created_s)/3600.0 AS ttc_h,
                   {_taker_pnl()} * count AS pnl, count AS w
            FROM {source} WHERE close_s >= created_s
        )
        SELECT {TTC_CASE} AS ttc_bucket, bin_lo, sum(pnl)/sum(w) * 100 AS vwar_pp
        FROM base GROUP BY 1, 2
    """
    g = con.execute(q).df()
    return g.pivot(index='ttc_bucket', columns='bin_lo', values='vwar_pp').reindex(TTC_ORDER)


def vwar_by_category(con, source, categorize_fn=None):
    """Volume-weighted taker/maker return (pp) per broad category, over ALL trades.
    Applies categorize to the DISTINCT prefixes only, then joins (fast at scale)."""
    fn = categorize_fn or categorize
    sm = con.execute(f"SELECT DISTINCT series FROM {source}").df()
    sm['category'] = sm['series'].map(fn)
    con.register('series_map', sm)
    q = f"""
        SELECT s.category,
               sum({_taker_pnl('t')} * t.count) / sum(t.count) * 100 AS taker_vwar_pp,
               sum(t.count) AS contracts, count(*) AS n
        FROM {source} t JOIN series_map s USING (series)
        GROUP BY 1 ORDER BY contracts DESC
    """
    o = con.execute(q).df()
    o['maker_vwar_pp'] = -o['taker_vwar_pp']
    return o


def aggregate_vwar(con, source):
    """Single-number maker/taker VWAR over all trades (paper Table II headline)."""
    q = f"SELECT sum({_taker_pnl()}*count)/sum(count)*100 AS taker_pp, sum(count) AS contracts FROM {source}"
    r = con.execute(q).df().iloc[0]
    return {'taker_vwar_pp': float(r['taker_pp']), 'maker_vwar_pp': -float(r['taker_pp']),
            'contracts': int(r['contracts'])}


def fee_comparison(con, source, categorize_fn=None):
    """Maker & taker VWAR (pp) GROSS and NET of Kalshi fees, overall ('All') + per
    category. Per-contract fees: taker 0.07*p*(1-p), maker 0.0175*p*(1-p). Net of fees
    maker != -taker (both pay the house), so each side is computed directly. Columns
    *_fee_pp are the volume-weighted average fee drag (gross - net)."""
    _series_map(con, source, categorize_fn)
    sel = (f"sum({_taker_pnl('t')}*t.count)/sum(t.count)*100 AS taker_gross, "
           f"sum({_taker_pnl_net('t')}*t.count)/sum(t.count)*100 AS taker_net, "
           f"sum(-({_taker_pnl('t')})*t.count)/sum(t.count)*100 AS maker_gross, "
           f"sum({_maker_pnl_net('t')}*t.count)/sum(t.count)*100 AS maker_net, "
           f"sum(t.count) AS contracts")
    by = con.execute(f"SELECT s.category, {sel} FROM {source} t JOIN series_map s USING (series) "
                     f"GROUP BY 1").df()
    allrow = con.execute(f"SELECT 'All' AS category, {sel} FROM {source} t "
                         f"JOIN series_map s USING (series)").df()
    out = pd.concat([allrow, by.sort_values('contracts', ascending=False)], ignore_index=True)
    out['taker_fee_pp'] = out['taker_gross'] - out['taker_net']
    out['maker_fee_pp'] = out['maker_gross'] - out['maker_net']
    out['vol_share_pct'] = 100 * out['contracts'] / out.loc[out['category'] != 'All', 'contracts'].sum()
    return out[['category', 'taker_gross', 'taker_net', 'taker_fee_pp',
                'maker_gross', 'maker_net', 'maker_fee_pp', 'vol_share_pct', 'contracts']]


def _series_map(con, source, categorize_fn=None):
    """Register a series->category lookup (categorize runs only on distinct prefixes)."""
    fn = categorize_fn or categorize
    sm = con.execute(f"SELECT DISTINCT series FROM {source}").df()
    sm['category'] = sm['series'].map(fn)
    con.register('series_map', sm)


def category_table(con, source, categorize_fn=None):
    """Per-category MAKER VWAR vs estimated half-spread, with net alpha and a
    volume-weighted monthly t-test of H0: net alpha = 0 (maker VWAR == half-spread).
    (Taker VWAR is omitted: it is exactly -maker.) Two-sided p_net.
    """
    _series_map(con, source, categorize_fn)
    # Aggregate per-category point estimates
    agg = con.execute(f"""
        SELECT s.category,
               -sum({_taker_pnl('t')} * t.count) / sum(t.count) * 100 AS maker_vwar_pp,
               sum(t.count) AS contracts, count(*) AS n
        FROM {source} t JOIN series_map s USING (series)
        GROUP BY 1
    """).df()
    agg['vol_share_pct'] = 100 * agg['contracts'] / agg['contracts'].sum()
    sp = spread_by_category(con, source, categorize_fn)[['category', 'half_spread_pp']]
    out = agg.merge(sp, on='category', how='left')
    out['net_alpha_pp'] = out['maker_vwar_pp'] - out['half_spread_pp']

    # Monthly maker VWAR and half-spread per category -> net -> volume-weighted t-test
    mk = con.execute(f"""
        WITH base AS (SELECT s.category, strftime(to_timestamp(t.created_s), '%Y-%m') AS month,
                             {_taker_pnl('t')}*t.count AS pnl, t.count AS w
                      FROM {source} t JOIN series_map s USING (series))
        SELECT category, month, -sum(pnl)/sum(w)*100 AS maker_vwar_pp, sum(w) AS contracts
        FROM base GROUP BY 1, 2
    """).df()
    spm = con.execute(f"""
        WITH d AS (SELECT s.category, t.created_s,
                          abs(t.yes_price - lag(t.yes_price)
                              OVER (PARTITION BY t.ticker ORDER BY t.created_s)) AS dp
                   FROM {source} t JOIN series_map s USING (series))
        SELECT category, strftime(to_timestamp(created_s), '%Y-%m') AS month,
               0.5*avg(dp) AS half_spread_pp
        FROM d WHERE dp IS NOT NULL GROUP BY 1, 2
    """).df()
    m = mk.merge(spm, on=['category', 'month'], how='inner')
    m['net_pp'] = m['maker_vwar_pp'] - m['half_spread_pp']
    rows = []
    for cat, g in m.groupby('category'):
        t, p2 = _weighted_ttest(g['net_pp'].values, g['contracts'].values)
        rows.append({'category': cat, 't_net': t, 'p_net': p2,
                     'n_months': int(g['net_pp'].notna().sum())})
    out = out.merge(pd.DataFrame(rows), on='category', how='left')

    cols = ['category', 'maker_vwar_pp', 'half_spread_pp', 'net_alpha_pp',
            't_net', 'p_net', 'n_months', 'vol_share_pct', 'contracts', 'n']
    return out[cols].sort_values('contracts', ascending=False).reset_index(drop=True)


def spread_by_category(con, source, categorize_fn=None):
    """Effective half-spread proxy per category: 0.5 * mean |Δ yes_price| between
    consecutive trades in the same market (cents == pp), per the paper's Section V."""
    _series_map(con, source, categorize_fn)
    q = f"""
        WITH d AS (
            SELECT series, count AS cnt,
                   abs(yes_price - lag(yes_price)
                       OVER (PARTITION BY ticker ORDER BY created_s)) AS dp
            FROM {source}
        )
        SELECT s.category, 0.5 * avg(d.dp) AS half_spread_pp,
               count(d.dp) AS n_pairs, sum(d.cnt) AS contracts
        FROM d JOIN series_map s USING (series)
        WHERE d.dp IS NOT NULL
        GROUP BY 1 ORDER BY contracts DESC
    """
    return con.execute(q).df()


def _weighted_ttest(x, w):
    """One-sample weighted t-test of mean(x) != 0 using Kish effective sample size.
    Returns (t, two_sided_p)."""
    x = np.asarray(x, float); w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if x.size < 2:
        return float('nan'), float('nan')
    wmean = np.sum(w * x) / np.sum(w)
    wvar = np.sum(w * (x - wmean) ** 2) / np.sum(w)
    n_eff = (np.sum(w) ** 2) / np.sum(w ** 2)        # Kish effective sample size
    if n_eff <= 1 or wvar <= 0:
        return float('nan'), float('nan')
    t = wmean / np.sqrt(wvar / n_eff)
    return float(t), float(2 * stats.t.sf(abs(t), n_eff - 1))


def _one_sided(t, p_two):
    """One-sided p (H1: mean > 0) from a two-sided p and the sign of t."""
    if not np.isfinite(t):
        return float('nan')
    return p_two / 2 if t > 0 else 1 - p_two / 2


def spread_compensation(con, source):
    """Maker VWAR vs estimated half-spread (paper Section V / Fig. 7).
    Half-spread = 0.5 * mean |Δ yes_price| between consecutive same-market trades.
    Monthly paired test, H1: maker VWAR > half-spread, reported BOTH equal-weighted
    (each month = 1 obs) and volume-weighted (months weighted by contract count).
    Returns (summary_dict, monthly_DataFrame)."""
    maker = con.execute(
        f"SELECT strftime(to_timestamp(created_s), '%Y-%m') AS month, "
        f"       -sum({_taker_pnl()}*count)/sum(count)*100 AS maker_vwar_pp, "
        f"       sum(count) AS contracts "
        f"FROM {source} GROUP BY 1").df()
    spread = con.execute(
        f"WITH d AS (SELECT created_s, abs(yes_price - lag(yes_price) "
        f"           OVER (PARTITION BY ticker ORDER BY created_s)) AS dp FROM {source}) "
        f"SELECT strftime(to_timestamp(created_s), '%Y-%m') AS month, 0.5*avg(dp) AS half_spread_pp "
        f"FROM d WHERE dp IS NOT NULL GROUP BY 1").df()
    m = maker.merge(spread, on='month').sort_values('month').reset_index(drop=True)
    m['net_pp'] = m['maker_vwar_pp'] - m['half_spread_pp']
    net = m.dropna(subset=['net_pp'])

    t_eq, p_eq2 = stats.ttest_1samp(net['net_pp'], 0.0)
    t_vw, p_vw2 = _weighted_ttest(net['net_pp'].values, net['contracts'].values)

    agg = aggregate_vwar(con, source)
    hs = con.execute(
        f"WITH d AS (SELECT abs(yes_price - lag(yes_price) "
        f"           OVER (PARTITION BY ticker ORDER BY created_s)) AS dp FROM {source}) "
        f"SELECT 0.5*avg(dp) FROM d WHERE dp IS NOT NULL").fetchone()[0]
    summary = {'maker_vwar_pp': agg['maker_vwar_pp'], 'half_spread_pp': float(hs),
               'net_alpha_pp': agg['maker_vwar_pp'] - float(hs),
               't_equal': float(t_eq), 'p_equal_1s': _one_sided(t_eq, p_eq2),
               't_volume': float(t_vw), 'p_volume_1s': _one_sided(t_vw, p_vw2),
               'n_months': int(net.shape[0])}
    return summary, m


def _daily_net(con, source):
    """Daily maker VWAR and half-spread -> daily net-alpha series (pp)."""
    maker = con.execute(
        f"SELECT to_timestamp(created_s)::DATE AS day, "
        f"       -sum({_taker_pnl()}*count)/sum(count)*100 AS maker_vwar_pp, sum(count) AS contracts "
        f"FROM {source} GROUP BY 1").df()
    spread = con.execute(
        f"WITH d AS (SELECT created_s, abs(yes_price - lag(yes_price) "
        f"           OVER (PARTITION BY ticker ORDER BY created_s)) AS dp FROM {source}) "
        f"SELECT to_timestamp(created_s)::DATE AS day, 0.5*avg(dp) AS half_spread_pp "
        f"FROM d WHERE dp IS NOT NULL GROUP BY 1").df()
    m = maker.merge(spread, on='day', how='inner').sort_values('day').reset_index(drop=True)
    m['net_pp'] = m['maker_vwar_pp'] - m['half_spread_pp']
    return m


def _newey_west_se(x, lag):
    """Newey-West (Bartlett-kernel) HAC standard error of the mean of series x."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float('nan'), n
    e = x - x.mean()
    lrv = np.dot(e, e) / n                                  # gamma_0
    for k in range(1, min(lag, n - 1) + 1):
        gk = np.dot(e[k:], e[:-k]) / n                      # autocovariance at lag k
        lrv += 2 * (1 - k / (lag + 1)) * gk                 # Bartlett weight
    lrv = max(lrv, 1e-12)
    return float(np.sqrt(lrv / n)), n                       # SE of the mean = sqrt(LRV/n)


def hac_spread_test(con, source, rolling=None, lag=None):
    """Test mean net-alpha (maker VWAR - half-spread) != 0 with naive vs Newey-West
    HAC standard errors. Set rolling=30 to reproduce the paper's overlapping 30-day
    series; HAC then corrects the autocorrelation the naive SE ignores.
    Returns a one-row dict."""
    s = _daily_net(con, source)['net_pp'].astype(float)
    label = 'daily (non-overlapping)'
    if rolling:
        s = s.rolling(rolling).mean().dropna()
        label = f'{rolling}d rolling (overlapping)'
    x = s.values
    n = x.size
    mean = float(np.mean(x)); sd = float(np.std(x, ddof=1))
    naive_se = sd / np.sqrt(n)
    if lag is None:                                         # default: window if rolling, else NW rule
        lag = rolling if rolling else int(np.floor(4 * (n / 100) ** (2 / 9)))
    hac_se, _ = _newey_west_se(x, lag)
    naive_t, hac_t = mean / naive_se, mean / hac_se
    return {'series': label, 'n': int(n), 'lag': int(lag), 'mean_net_pp': mean,
            'naive_se': float(naive_se), 'naive_t': float(naive_t),
            'naive_p': float(2 * stats.norm.sf(abs(naive_t))),
            'hac_se': float(hac_se), 'hac_t': float(hac_t),
            'hac_p': float(2 * stats.norm.sf(abs(hac_t)))}


def _weighted_nw(net, w, lag):
    """Weighted mean of `net` (weights `w`) with naive and Bartlett HAC SEs.
    Returns (theta, naive_se, hac_se). w=ones -> ordinary mean + Newey-West."""
    net = np.asarray(net, float); w = np.asarray(w, float)
    ok = np.isfinite(net) & np.isfinite(w) & (w > 0)
    net, w = net[ok], w[ok]
    n = net.size
    if n < 5:
        return float('nan'), float('nan'), float('nan')
    a = w / w.sum()                                     # normalized weights
    theta = float(np.sum(a * net))
    e = net - theta
    v0 = np.sum((a * e) ** 2)                            # naive var of the weighted mean
    v = v0
    for k in range(1, min(lag, n - 1) + 1):             # Bartlett-weighted autocovariances
        v += 2 * (1 - k / (lag + 1)) * np.sum(a[k:] * a[:-k] * e[k:] * e[:-k])
    return theta, float(np.sqrt(max(v0, 1e-18))), float(np.sqrt(max(v, 1e-18)))


def _rolling_net_tests(g, window=30):
    """From a per-group daily frame (maker_vwar_pp, half_spread_pp, w) build the rolling
    net series and test it BOTH equal-weighted and volume-weighted (weights = window
    volume), each with paper-style naive and Newey-West HAC t (two-sided p, lag=window)."""
    g = g.sort_values('dt')
    net = (g['maker_vwar_pp'] - g['half_spread_pp']).rolling(window).mean()
    vol = g['w'].rolling(window).sum()
    ok = net.notna() & vol.notna()
    net, vol = net[ok].values, vol[ok].values
    sf = lambda t: float(2 * stats.norm.sf(abs(t))) if np.isfinite(t) else float('nan')
    out = {'n_points': int(net.size)}
    for tag, wts in (('eq', np.ones_like(net)), ('vw', vol)):
        th, se0, seh = _weighted_nw(net, wts, window)
        out[f'{tag}_net'] = th
        out[f'{tag}_naive_t'] = th / se0; out[f'{tag}_naive_p'] = sf(th / se0)
        out[f'{tag}_hac_t'] = th / seh;   out[f'{tag}_hac_p'] = sf(th / seh)
    return out


def spread_hac_by_category(con, source, window=30, categorize_fn=None):
    """Per category (+ 'All'): rolling net alpha (maker VWAR - half-spread), tested
    EQUAL-weighted and VOLUME-weighted, each with the paper's naive overlapping-window
    t and a Newey-West HAC t (lag=window). Two-sided p-values."""
    _series_map(con, source, categorize_fn)
    mk = con.execute(f"""
        SELECT s.category, to_timestamp(t.created_s)::DATE AS dt,
               -sum({_taker_pnl('t')}*t.count)/sum(t.count)*100 AS maker_vwar_pp,
               sum(t.count) AS w
        FROM {source} t JOIN series_map s USING (series) GROUP BY 1, 2""").df()
    sp = con.execute(f"""
        WITH d AS (SELECT s.category, t.created_s,
                          abs(t.yes_price - lag(t.yes_price)
                              OVER (PARTITION BY t.ticker ORDER BY t.created_s)) AS dp
                   FROM {source} t JOIN series_map s USING (series))
        SELECT category, to_timestamp(created_s)::DATE AS dt, 0.5*avg(dp) AS half_spread_pp
        FROM d WHERE dp IS NOT NULL GROUP BY 1, 2""").df()
    per = mk.merge(sp, on=['category', 'dt'], how='inner')

    # 'All' = pool categories per day (volume-weighted maker & half-spread, total volume)
    a = (per.assign(pnl=per.maker_vwar_pp * per.w, hsw=per.half_spread_pp * per.w)
            .groupby('dt').agg(pnl=('pnl', 'sum'), hsw=('hsw', 'sum'), w=('w', 'sum')).reset_index())
    a['maker_vwar_pp'] = a.pnl / a.w; a['half_spread_pp'] = a.hsw / a.w; a['category'] = 'All'
    allrows = a[['category', 'dt', 'maker_vwar_pp', 'half_spread_pp', 'w']]
    full = pd.concat([per[['category', 'dt', 'maker_vwar_pp', 'half_spread_pp', 'w']], allrows])

    rows = []
    for cat, g in full.groupby('category'):
        rows.append({'category': cat, **_rolling_net_tests(g, window)})
    out = pd.DataFrame(rows)
    order = ['All'] + sorted(c for c in out['category'] if c != 'All')
    return out.set_index('category').loc[order].reset_index()


def data_summary(con, source, categorize_fn=None):
    """Simple data breakdown by category: # markets, # trades, contract volume, %YES,
    and median/mean TRADE time-to-close in hours = (close_s - created_s) -- the same
    quantity used in the delta facets. Appends an 'All' total row."""
    _series_map(con, source, categorize_fn)
    ttc = "CASE WHEN t.close_s >= t.created_s THEN (t.close_s - t.created_s)/3600.0 END"
    by_cat = con.execute(f"""
        SELECT s.category, count(DISTINCT t.ticker) AS markets, count(*) AS trades,
               sum(t.count) AS contracts, 100.0*avg((t.result='yes')::INT) AS pct_yes,
               median({ttc}) AS med_ttc_h, avg({ttc}) AS mean_ttc_h
        FROM {source} t JOIN series_map s USING (series)
        GROUP BY 1 ORDER BY contracts DESC
    """).df()
    ttc0 = ttc.replace('t.', '')
    tot = con.execute(f"""
        SELECT count(DISTINCT ticker) AS markets, count(*) AS trades, sum(count) AS contracts,
               100.0*avg((result='yes')::INT) AS pct_yes,
               median({ttc0}) AS med_ttc_h, avg({ttc0}) AS mean_ttc_h
        FROM {source}
    """).df()
    tot.insert(0, 'category', 'All')
    return pd.concat([by_cat, tot], ignore_index=True)


def yes_no_share(con, source, categorize_fn=None):
    """Affirmative tilt: YES share of taker contracts, capital, and trade count,
    overall and per category (paper eq. 9 / Fig. 11). Appends an 'All' row."""
    _series_map(con, source, categorize_fn)
    yes_cap = "CASE WHEN t.taker_side='yes' THEN t.count*t.yes_price/100.0 ELSE 0 END"
    no_cap = "CASE WHEN t.taker_side='no' THEN t.count*(100.0 - t.yes_price)/100.0 ELSE 0 END"
    by_cat = con.execute(f"""
        SELECT s.category,
               100.0*sum(CASE WHEN t.taker_side='yes' THEN t.count ELSE 0 END)/sum(t.count) AS yes_vol_pct,
               100.0*sum({yes_cap})/(sum({yes_cap}) + sum({no_cap})) AS yes_capital_pct,
               100.0*avg((t.taker_side='yes')::INT) AS yes_trade_pct,
               sum(t.count) AS contracts, count(*) AS n
        FROM {source} t JOIN series_map s USING (series)
        GROUP BY 1 ORDER BY contracts DESC
    """).df()
    yes_cap0 = yes_cap.replace('t.', '')
    no_cap0 = no_cap.replace('t.', '')
    tot = con.execute(f"""
        SELECT 100.0*sum(CASE WHEN taker_side='yes' THEN count ELSE 0 END)/sum(count) AS yes_vol_pct,
               100.0*sum({yes_cap0})/(sum({yes_cap0}) + sum({no_cap0})) AS yes_capital_pct,
               100.0*avg((taker_side='yes')::INT) AS yes_trade_pct,
               sum(count) AS contracts, count(*) AS n FROM {source}
    """).df()
    tot.insert(0, 'category', 'All')
    return pd.concat([by_cat, tot], ignore_index=True)


def vwar_by_side(con, source):
    """Per price bucket and taker side: that side's own VWAR (pp), volume, trades.
    YES-taker return = y - p; NO-taker return = p - y."""
    q = f"""
        SELECT (yes_price // {PRICE_BIN}) * {PRICE_BIN} / 100.0 AS bin_lo, taker_side,
               sum({_taker_pnl()}*count)/sum(count)*100 AS vwar_pp,
               sum(count) AS contracts, count(*) AS n
        FROM {source} GROUP BY 1, 2 ORDER BY 1, 2
    """
    d = con.execute(q).df()
    d['mid'] = d['bin_lo'] + PRICE_BIN / 200.0
    return d


def vwar_side_grid(con, source, taker_side='yes'):
    """Taker VWAR (pp) on the time-to-close x price grid for ONE taker side."""
    q = f"""
        WITH base AS (
            SELECT (yes_price // {PRICE_BIN}) * {PRICE_BIN} / 100.0 AS bin_lo,
                   (close_s - created_s)/3600.0 AS ttc_h,
                   {_taker_pnl()}*count AS pnl, count AS w
            FROM {source} WHERE taker_side = '{taker_side}' AND close_s >= created_s
        )
        SELECT {TTC_CASE} AS ttc_bucket, bin_lo, sum(pnl)/sum(w)*100 AS vwar_pp
        FROM base GROUP BY 1, 2
    """
    g = con.execute(q).df()
    return g.pivot(index='ttc_bucket', columns='bin_lo', values='vwar_pp').reindex(TTC_ORDER)


def vwar_category_ttc(con, source, role='maker', categorize_fn=None):
    """Volume-weighted VWAR (pp) per (category x time-to-close bucket).
    role='maker' (default) or 'taker'. Returns a category-by-ttc pivot (pp)."""
    _series_map(con, source, categorize_fn)
    sign = '-' if role == 'maker' else ''
    q = f"""
        WITH base AS (
            SELECT s.category, (t.close_s - t.created_s)/3600.0 AS ttc_h,
                   {_taker_pnl('t')}*t.count AS pnl, t.count AS w
            FROM {source} t JOIN series_map s USING (series)
            WHERE t.close_s >= t.created_s
        )
        SELECT category, {TTC_CASE} AS ttc_bucket, {sign}sum(pnl)/sum(w)*100 AS vwar_pp,
               sum(w) AS contracts
        FROM base GROUP BY 1, 2
    """
    g = con.execute(q).df()
    order = (g.groupby('category')['contracts'].sum().sort_values(ascending=False).index)
    return (g.pivot(index='category', columns='ttc_bucket', values='vwar_pp')
             .reindex(index=order, columns=TTC_ORDER))


def calibration_by_category(con, source, side='yes', categorize_fn=None):
    """Calibration delta_b per (category x price bucket), taker-side, with t-test.
    Replicates the FLSB delta panel separately for each category."""
    _series_map(con, source, categorize_fn)
    q = f"""
        WITH base AS (
            SELECT s.category, (t.yes_price // {PRICE_BIN})*{PRICE_BIN}/100.0 AS bin_lo,
                   (t.result='yes')::INT - t.yes_price/100.0 AS dev, t.count AS w
            FROM {source} t JOIN series_map s USING (series)
            WHERE t.taker_side = '{side}'
        )
        SELECT category, bin_lo, count(*) AS n, sum(w) AS contracts,
               avg(dev) AS delta, stddev_samp(dev) AS sd_dev
        FROM base GROUP BY 1, 2 ORDER BY 1, 2
    """
    d = _add_ttest(con.execute(q).df())
    d['mid'] = d['bin_lo'] + PRICE_BIN / 200.0
    return d


def vwar_side_by_category(con, source, categorize_fn=None):
    """Per (category x price bucket x taker side): that side's own VWAR (pp).
    Figure 9 (YES@p vs NO@p taker returns) decomposed by category."""
    _series_map(con, source, categorize_fn)
    q = f"""
        SELECT s.category, (t.yes_price // {PRICE_BIN})*{PRICE_BIN}/100.0 AS bin_lo, t.taker_side,
               sum({_taker_pnl('t')}*t.count)/sum(t.count)*100 AS vwar_pp,
               sum(t.count) AS contracts, count(*) AS n
        FROM {source} t JOIN series_map s USING (series)
        GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
    """
    d = con.execute(q).df()
    d['mid'] = d['bin_lo'] + PRICE_BIN / 200.0
    return d


# ---------------------------------------------------------------------------
# Plotting (each returns nothing; saves to `save` if given, else plt.show())
# ---------------------------------------------------------------------------
def _emph(df, values_pp, sig_col='pval', thresh=0.0):
    """Boolean 'emphasize' mask: significant (p<0.05) or |effect| >= thresh.
    NOTE: color encodes MAGNITUDE/significance, not sign. Light bars are below the
    threshold (small effect) or non-significant -- a small negative bar is light."""
    if sig_col and sig_col in df:
        return (df[sig_col] < 0.05).values
    return (np.abs(values_pp) >= thresh).values


def _emph_legend(ax, sig_col, thresh):
    """Legend explaining the dark/light bar coloring."""
    if sig_col:
        items = [('#1f77b4', 'p < 0.05'), ('#9ecae1', 'p ≥ 0.05')]
    else:
        items = [('#1f77b4', f'|effect| ≥ {thresh:g}pp'), ('#9ecae1', f'|effect| < {thresh:g}pp')]
    ax.legend(handles=[Patch(color=c, label=l) for c, l in items], loc='best', fontsize=8)


def _finish(fig, save, dpi=130):
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"saved {save}")
    else:
        plt.show()


def plot_calibration(cal, title='Kalshi Favorite-Longshot Bias', sig_col='pval',
                     thresh=0.0, save=None):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    sizes = 40 + 300 * (cal['n'] / cal['n'].max())
    axL.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
    axL.scatter(cal['pbar'], cal['f'], s=sizes, c=cal['mid'], cmap='viridis',
                edgecolor='k', linewidth=0.5, zorder=3)
    axL.axvspan(0, LONGSHOT, color='red', alpha=0.06)
    axL.axvspan(FAVORITE, 1.0, color='green', alpha=0.06)
    axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.set_aspect('equal')
    axL.set_xlabel('Implied probability  P̄_b'); axL.set_ylabel('Empirical YES win rate  f(b)')
    axL.set_title('Calibration curve — taker-YES bets'); axL.legend(loc='upper left')

    delta_pp = cal['delta'] * 100
    colors = ['#1f77b4' if e else '#9ecae1' for e in _emph(cal, delta_pp, sig_col, thresh)]
    axR.bar(cal['mid'], delta_pp, width=0.04, color=colors, edgecolor='k', linewidth=0.4)
    axR.axhline(0, color='k', lw=0.8); axR.axvspan(0, LONGSHOT, color='red', alpha=0.06)
    axR.set_xlim(0, 1)
    axR.set_xlabel('Price bucket midpoint'); axR.set_ylabel('δ_b = f(b) − P̄_b  (pp)')
    axR.set_title('Calibration deviation δ_b by bucket'); _emph_legend(axR, sig_col, thresh)
    fig.suptitle(title, fontweight='bold'); _finish(fig, save)


def plot_delta_facets(d, title='δ_b by price bucket, faceted by time-to-close',
                      sig_col='pval', thresh=0.0, sharey=False, save=None):
    """sharey=False (default) lets each time-to-close panel auto-scale to its own
    delta range; pass sharey=True for a common scale."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=sharey)
    for ax, lab in zip(axes.ravel(), TTC_ORDER):
        sub = d[d['ttc_bucket'] == lab]
        if len(sub) == 0:
            ax.set_title(f'{lab}  (no trades)'); continue
        delta_pp = sub['delta'] * 100
        colors = ['#1f77b4' if e else '#9ecae1' for e in _emph(sub, delta_pp, sig_col, thresh)]
        ax.bar(sub['mid'], delta_pp, width=0.04, color=colors, edgecolor='k', linewidth=0.4)
        ax.axhline(0, color='k', lw=0.8); ax.axvspan(0, LONGSHOT, color='red', alpha=0.06)
        ax.set_xlim(0, 1); ax.set_title(f"{lab}   (n={int(sub['n'].sum()):,})")
    _emph_legend(axes.ravel()[0], sig_col, thresh)
    for ax in axes[-1]:
        ax.set_xlabel('Price bucket midpoint')
    for ax in axes[:, 0]:
        ax.set_ylabel('δ_b (pp)')
    fig.suptitle(title, fontweight='bold'); _finish(fig, save)


def plot_deltatime(reg, title='Calibration deviation vs. time-to-close, per price bucket',
                   sig_col='pval', thresh=0.0, save=None):
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ['#1f77b4' if e else '#9ecae1'
              for e in _emph(reg, reg['slope_pp_day'], sig_col, thresh)]
    ax.bar(reg['mid'], reg['slope_pp_day'], width=0.04, color=colors, edgecolor='k', linewidth=0.4)
    ax.axhline(0, color='k', lw=0.8); ax.axvspan(0, LONGSHOT, color='red', alpha=0.06)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Price bucket midpoint'); ax.set_ylabel('δ change per day to close (pp/day)')
    _emph_legend(ax, sig_col, thresh)
    ax.set_title(title, fontweight='bold'); _finish(fig, save)


def plot_vwar(pb, grid, title='Taker volume-weighted returns (all trades)', save=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5.5),
                                   gridspec_kw={'width_ratios': [1, 1.4]})
    colors = ['#2ca02c' if x >= 0 else '#d62728' for x in pb['vwar_pp']]
    ax1.bar(pb['mid'], pb['vwar_pp'], width=0.04, color=colors, edgecolor='k', linewidth=0.4)
    ax1.axhline(0, color='k', lw=0.8); ax1.axvspan(0, LONGSHOT, color='red', alpha=0.06)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Price bucket midpoint'); ax1.set_ylabel('Taker VWAR (pp)')
    ax1.set_title('VWAR by price bucket')

    vmax = np.nanmax(np.abs(grid.values)) if np.isfinite(grid.values).any() else 1.0
    im = ax2.imshow(grid.values, aspect='auto', cmap='RdYlGn', vmin=-vmax, vmax=vmax)
    ax2.set_xticks(range(len(grid.columns)))
    ax2.set_xticklabels([f'{c:.2f}' for c in grid.columns], rotation=90, fontsize=8)
    ax2.set_yticks(range(len(grid.index))); ax2.set_yticklabels(grid.index)
    ax2.set_xlabel('Price bucket midpoint'); ax2.set_ylabel('Time to close')
    ax2.set_title('VWAR (pp): time-to-close x price bucket')
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid.values[i, j]
            if not np.isnan(val):
                ax2.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6)
    fig.colorbar(im, ax=ax2, label='Taker VWAR (pp)')
    fig.suptitle(title, fontweight='bold'); _finish(fig, save)


def plot_category(cat, title='Maker VWAR vs. half-spread by category', save=None):
    c = cat.sort_values('contracts')                      # smallest at top -> biggest at bottom
    y = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(c) + 1.6))
    ax.barh(y, c['maker_vwar_pp'], height=0.55, color='#2ca02c', label='Maker VWAR')
    ax.scatter(c['half_spread_pp'], y, color='k', marker='|', s=400, zorder=3,
               label='Est. half-spread')
    if 'p_net' in c:                                      # mark net alpha significantly != 0
        for yi, (mk, pv) in enumerate(zip(c['maker_vwar_pp'], c['p_net'])):
            if np.isfinite(pv) and pv < 0.05:
                ax.text(mk, yi, ' *', va='center', ha='left', fontsize=13, fontweight='bold')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(c['category'])
    ax.set_xlabel('Volume-weighted return (pp)   (bar=maker VWAR, tick=half-spread, *=net≠0 p<0.05)')
    ax.set_title(title, fontweight='bold'); ax.legend(loc='lower right')
    _finish(fig, save)


def plot_fee_comparison(fc, title='Maker & Taker VWAR: gross vs. net of fees', save=None):
    c = fc[fc['category'] != 'All'].sort_values('contracts')
    y = np.arange(len(c)); h = 0.38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 0.55 * len(c) + 2), sharey=True)
    a1.barh(y + h/2, c['maker_gross'], height=h, color='#9ecae1', label='gross')
    a1.barh(y - h/2, c['maker_net'],   height=h, color='#2ca02c', label='net of fees')
    a1.axvline(0, color='k', lw=0.8); a1.set_title('Maker VWAR'); a1.set_xlabel('pp'); a1.legend()
    a1.set_yticks(y); a1.set_yticklabels(c['category'])
    a2.barh(y + h/2, c['taker_gross'], height=h, color='#fdae6b', label='gross')
    a2.barh(y - h/2, c['taker_net'],   height=h, color='#d62728', label='net of fees')
    a2.axvline(0, color='k', lw=0.8); a2.set_title('Taker VWAR'); a2.set_xlabel('pp'); a2.legend()
    fig.suptitle(title, fontweight='bold'); _finish(fig, save)


def plot_data_summary(d, title='Data breakdown by category', save=None):
    c = d[d['category'] != 'All'].sort_values('contracts')
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.barh(c['category'], c['contracts'], color='#4c78a8')
    a1.set_xlabel('Contract volume'); a1.set_title('Volume by category')
    a2.barh(c['category'], c['med_ttc_h'], color='#54a24b')
    a2.set_xlabel('Median trade time-to-close (h)'); a2.set_title('Median time-to-close')
    fig.suptitle(title, fontweight='bold'); _finish(fig, save)


def plot_yes_no_share(d, share_col='yes_vol_pct',
                      title='YES taker contract-volume share by category',
                      xlabel='YES share of taker contract volume (%)', save=None):
    c = d[d['category'] != 'All'].sort_values(share_col)
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(c) + 1.5))
    ax.barh(c['category'], c[share_col], color='#2ca02c')
    ax.axvline(50, color='k', ls='--', lw=1, label='50% neutral')
    ax.set_xlabel(xlabel); ax.legend(loc='lower right')
    ax.set_title(title, fontweight='bold'); _finish(fig, save)


def plot_side_price(d, title='Taker VWAR by side & price bucket', save=None):
    piv = d.pivot(index='mid', columns='taker_side', values='vwar_pp').sort_index()
    x = piv.index.values; w = 0.02
    fig, ax = plt.subplots(figsize=(12, 5))
    if 'yes' in piv:
        ax.bar(x - w / 2, piv['yes'], width=w, color='#2ca02c', label='YES taker')
    if 'no' in piv:
        ax.bar(x + w / 2, piv['no'], width=w, color='#d62728', label='NO taker')
    ax.axhline(0, color='k', lw=0.8); ax.axvspan(0, LONGSHOT, color='red', alpha=0.06)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Price bucket midpoint'); ax.set_ylabel('Taker VWAR (pp)')
    ax.legend(); ax.set_title(title, fontweight='bold'); _finish(fig, save)


def plot_side_grids(grid_yes, grid_no,
                    title='Taker VWAR (pp): time-to-close x price, by side', save=None):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5.5))
    vmax = max((np.nanmax(np.abs(g.values)) if np.isfinite(g.values).any() else 1.0)
               for g in (grid_yes, grid_no))
    for ax, g, lab in zip(axes, (grid_yes, grid_no), ('YES taker', 'NO taker')):
        im = ax.imshow(g.values, aspect='auto', cmap='RdYlGn', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(g.columns)))
        ax.set_xticklabels([f'{c:.2f}' for c in g.columns], rotation=90, fontsize=7)
        ax.set_yticks(range(len(g.index))); ax.set_yticklabels(g.index)
        ax.set_xlabel('Price bucket midpoint'); ax.set_ylabel('Time to close'); ax.set_title(lab)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5)
        fig.colorbar(im, ax=ax, label='VWAR (pp)')
    fig.suptitle(title, fontweight='bold'); _finish(fig, save)


def plot_category_ttc(piv, role='maker', save=None):
    """Heatmap of VWAR (pp) over category (rows) x time-to-close (cols)."""
    vals = piv.values.astype(float)
    vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(piv) + 2))
    im = ax.imshow(vals, aspect='auto', cmap='RdYlGn', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    ax.set_xlabel('Time to close'); ax.set_ylabel('Category')
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = vals[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, label=f'{role.capitalize()} VWAR (pp)')
    ax.set_title(f'{role.capitalize()} VWAR (pp): category x time-to-close',
                 fontweight='bold'); _finish(fig, save)


def _facet_axes(cats, ncols=3, panel=(5, 3.2), sharey=False):
    """Create a category-faceted subplot grid; returns (fig, {category: ax}).
    sharey defaults False so each category auto-scales to its own effect size."""
    nrows = int(np.ceil(len(cats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows),
                             sharex=True, sharey=sharey, squeeze=False)
    axf = axes.ravel()
    for ax in axf[len(cats):]:
        ax.set_visible(False)
    return fig, dict(zip(cats, axf))


def plot_calibration_by_category(d, thresh=1.0, sig_col=None, ncols=3, sharey=False, save=None):
    """Faceted FLSB delta_b bars: one panel per category (ordered by volume).
    sharey=False (default) lets each category auto-scale to its own delta range."""
    cats = list(d.groupby('category')['n'].sum().sort_values(ascending=False).index)
    fig, axmap = _facet_axes(cats, ncols, sharey=sharey)
    for cat, ax in axmap.items():
        sub = d[d['category'] == cat]
        dp = sub['delta'] * 100
        colors = ['#1f77b4' if e else '#9ecae1' for e in _emph(sub, dp, sig_col, thresh)]
        ax.bar(sub['mid'], dp, width=0.04, color=colors, edgecolor='k', linewidth=0.3)
        ax.axhline(0, color='k', lw=0.8); ax.axvspan(0, LONGSHOT, color='red', alpha=0.06)
        ax.set_xlim(0, 1); ax.set_title(f"{cat}  (n={int(sub['n'].sum()):,})", fontsize=9)
    fig.suptitle('Calibration δ_b by price bucket, per category', fontweight='bold')
    fig.supxlabel('Price bucket midpoint'); fig.supylabel('δ_b = f(b) − P̄_b  (pp)')
    _finish(fig, save)


def plot_side_by_category(d, ncols=3, sharey=False, save=None):
    """Faceted YES@p vs NO@p taker VWAR (Fig. 9): one panel per category.
    sharey=False (default) lets each category auto-scale to its own VWAR range."""
    cats = list(d.groupby('category')['contracts'].sum().sort_values(ascending=False).index)
    fig, axmap = _facet_axes(cats, ncols, sharey=sharey)
    first = True
    for cat, ax in axmap.items():
        piv = (d[d['category'] == cat].pivot(index='mid', columns='taker_side', values='vwar_pp')
               .sort_index())
        x = piv.index.values; w = 0.02
        if 'yes' in piv:
            ax.bar(x - w / 2, piv['yes'], width=w, color='#2ca02c', label='YES taker')
        if 'no' in piv:
            ax.bar(x + w / 2, piv['no'], width=w, color='#d62728', label='NO taker')
        ax.axhline(0, color='k', lw=0.8); ax.axvspan(0, LONGSHOT, color='red', alpha=0.06)
        ax.set_xlim(0, 1); ax.set_title(cat, fontsize=9)
        if first:
            ax.legend(fontsize=8); first = False
    fig.suptitle('Taker VWAR: YES vs NO by price bucket, per category', fontweight='bold')
    fig.supxlabel('Price bucket midpoint'); fig.supylabel('Taker VWAR (pp)')
    _finish(fig, save)


# ---------------------------------------------------------------------------
# Orchestration: run every analysis and save all figures
# ---------------------------------------------------------------------------
def run_all(con, source, outdir='results', thresh=1.0):
    """Compute every analysis from `source`, saving CSVs to <outdir>/tables and figures
    to <outdir>/plots. Returns a dict of the computed tables; `thresh` colors bars by
    |effect| >= thresh."""
    tables_dir = os.path.join(outdir, 'tables')
    plots_dir = os.path.join(outdir, 'plots')
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    def p(name):                                        # route by extension
        return os.path.join(plots_dir if name.endswith('.png') else tables_dir, name)

    # Table-I-style data breakdown (Fig. 1)
    summ = data_summary(con, source)
    summ.to_csv(p('data_summary.csv'), index=False)
    plot_data_summary(summ, save=p('data_summary.png'))

    agg = aggregate_vwar(con, source)
    print(f"AGGREGATE  maker VWAR = {agg['maker_vwar_pp']:+.3f} pp  "
          f"(taker {agg['taker_vwar_pp']:+.3f} pp)  over {agg['contracts']:,} contracts")

    # Returns gross vs net of Kalshi fees (taker 0.07*p(1-p), maker 0.0175*p(1-p))
    fee = fee_comparison(con, source)
    fee.to_csv(p('fee_comparison.csv'), index=False)
    plot_fee_comparison(fee, save=p('fee_comparison.png'))
    a = fee[fee['category'] == 'All'].iloc[0]
    print(f"FEES       maker {a['maker_gross']:+.3f}->{a['maker_net']:+.3f}pp (fee {a['maker_fee_pp']:.3f}) | "
          f"taker {a['taker_gross']:+.3f}->{a['taker_net']:+.3f}pp (fee {a['taker_fee_pp']:.3f})")

    # Spread-compensation test (paper Section V): both equal- and volume-weighted
    sc, _ = spread_compensation(con, source)
    print(f"SPREAD     maker {sc['maker_vwar_pp']:+.3f}pp vs half-spread {sc['half_spread_pp']:.3f}pp "
          f"-> net {sc['net_alpha_pp']:+.3f}pp | equal-wt t={sc['t_equal']:.2f} "
          f"(p1={sc['p_equal_1s']:.3g}) | vol-wt t={sc['t_volume']:.2f} "
          f"(p1={sc['p_volume_1s']:.3g}) | n={sc['n_months']} months")

    # HAC robustness: naive vs Newey-West SE on daily and (paper-style) 30d-rolling net
    hac = pd.DataFrame([hac_spread_test(con, source), hac_spread_test(con, source, rolling=30)])
    hac.to_csv(p('spread_hac.csv'), index=False)
    hac_cat = spread_hac_by_category(con, source)        # naive (paper) vs HAC, per category
    hac_cat.to_csv(p('spread_hac_by_category.csv'), index=False)
    for _, r in hac.iterrows():
        print(f"HAC[{r['series']}]  net {r['mean_net_pp']:+.3f}pp  "
              f"naive t={r['naive_t']:.1f} (p={r['naive_p']:.2g})  "
              f"HAC t={r['hac_t']:.1f} (p={r['hac_p']:.2g}, lag={int(r['lag'])})")

    cal = calibration_table(con, source)
    plot_calibration(cal, sig_col=None, thresh=thresh, save=p('calibration.png'))
    cal.to_csv(p('calibration.csv'), index=False)

    d = calibration_by_ttc(con, source)
    plot_delta_facets(d, sig_col=None, thresh=thresh, save=p('delta_facets.png'))

    reg = deltatime_regression(con, source)              # OLS table w/ p-values
    plot_deltatime(reg, save=p('deltatime.png'))
    reg.to_csv(p('deltatime_regression.csv'), index=False)

    pb = vwar_by_price(con, source)
    grid = vwar_grid(con, source)
    plot_vwar(pb, grid, save=p('vwar_price_time.png'))

    cat = category_table(con, source)                    # category VWAR + %vol + half-spread + sig
    plot_category(cat, save=p('vwar_by_category.png'))
    cat.to_csv(p('category_breakdown.csv'), index=False)

    yn = yes_no_share(con, source)                       # YES/NO tilt overall + per category
    yn.to_csv(p('yes_no_share.csv'), index=False)
    plot_yes_no_share(
        yn,
        share_col='yes_vol_pct',
        title='YES taker contract-volume share by category',
        xlabel='YES share of taker contract volume (%)',
        save=p('yes_no_share.png'),
    )
    plot_yes_no_share(
        yn,
        share_col='yes_vol_pct',
        title='YES taker contract-volume share by category',
        xlabel='YES share of taker contract volume (%)',
        save=p('yes_no_share_volume.png'),
    )
    plot_yes_no_share(
        yn,
        share_col='yes_capital_pct',
        title='YES taker capital share by category',
        xlabel='YES share of taker capital (%)',
        save=p('yes_no_share_capital.png'),
    )

    side = vwar_by_side(con, source)                     # YES/NO taker VWAR per price bucket
    side.to_csv(p('vwar_by_side.csv'), index=False)
    plot_side_price(side, save=p('vwar_by_side_price.png'))
    g_yes = vwar_side_grid(con, source, 'yes')           # ... and per time-to-close x price
    g_no = vwar_side_grid(con, source, 'no')
    plot_side_grids(g_yes, g_no, save=p('vwar_by_side_grid.png'))

    cat_ttc = vwar_category_ttc(con, source, role='maker')   # category x time-to-close VWAR
    cat_ttc.to_csv(p('vwar_category_ttc.csv'))
    plot_category_ttc(cat_ttc, role='maker', save=p('vwar_category_ttc.png'))

    cal_cat = calibration_by_category(con, source)           # FLSB delta per category
    cal_cat.to_csv(p('calibration_by_category.csv'), index=False)
    plot_calibration_by_category(cal_cat, thresh=thresh, save=p('calibration_by_category.png'))

    side_cat = vwar_side_by_category(con, source)            # Fig. 9 (YES vs NO) per category
    side_cat.to_csv(p('vwar_side_by_category.csv'), index=False)
    plot_side_by_category(side_cat, save=p('vwar_side_by_category.png'))

    print("tables: data_summary.csv, calibration.csv, deltatime_regression.csv, "
          "category_breakdown.csv, yes_no_share.csv, vwar_by_side.csv, vwar_category_ttc.csv, "
          "calibration_by_category.csv, vwar_side_by_category.csv")
    return {'data_summary': summ, 'aggregate': agg, 'fee_comparison': fee,
            'spread_compensation': sc, 'spread_hac': hac,
            'calibration': cal, 'calibration_by_ttc': d, 'deltatime': reg,
            'vwar_by_price': pb, 'vwar_grid': grid, 'category': cat,
            'yes_no_share': yn, 'vwar_by_side': side,
            'side_grid_yes': g_yes, 'side_grid_no': g_no, 'category_ttc': cat_ttc,
            'calibration_by_category': cal_cat, 'vwar_side_by_category': side_cat}
