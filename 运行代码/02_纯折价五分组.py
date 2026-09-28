# -*- coding: utf-8 -*-
# Author：@跟陈博士学AI
"""
规划 1.0 第 2 步：纯折价因子，月末截面五分组。

问：A 相对 H 溢价越高，未来 H 是否越好？
通过：高溢价组显著优于低溢价组，或五组收益至少单调。

口径（对齐 AH折溢价研究规划 1.0）：
  - 样本 2017-01-03 起
  - 信号日：每月最后一个「可新开仓」出现过的港股交易日
  - 可新开仓：trade_ah_tradable_daily.is_tradable=1
    （窗口内、H 未停牌、20 日成交额过固定门槛、两边不复权正价）
  - 排序：当日不复权 premium_pct，越高表示 H 越便宜，记为第 5 组
  - 成交：信号日之后的下一根 H 后复权收盘
  - 收益：后复权收盘比，持有 21 / 63 / 126 个该股交易日（约 1 / 3 / 6 个月）
  - 窗口中途结束：收益算到 pair_end 上最后一根后复权收盘，不使用窗口外价格
  - 样本末端还凑不满持有期：该期限收益记空，不外推
  - 海通换股不在这一步把折价序列接到国泰君安；持有期越过 pair_end 就停在海通最后价

运行：
  python 02_纯折价五分组.py
  这一步连接本机 MySQL，重建月末信号。
  只复现分组和图：python 02_从信号CSV复现.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PKG = Path(__file__).resolve().parent.parent
DATA = PKG / '原始数据'
OUT = PKG / '课题研究结论产出'
DATA_SRC = Path(r'd:\JGY\zhihu\AI量化交易第1期\4-数据获取与清洗\CASE-数据采集')


def get_connection():
    """连接 MySQL。只有本文件直接重建信号时才调用，读 CSV 复现不会走到这里。"""
    sys.path.insert(0, str(DATA_SRC))
    from db_config import get_connection as _connect
    return _connect()

START = '2017-01-03'
HORIZONS = (('1m', 21), ('3m', 63), ('6m', 126))
# 持有期重叠时，Newey-West 滞后约等于「月份数 - 1」
NW_LAGS = {'1m': 0, '3m': 2, '6m': 5}
MIN_NAMES = 15

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def read_sql(sql: str, params=None) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()
    return df


def load_signals() -> pd.DataFrame:
    """月末可新开仓截面，带溢价、行业、窗口终点。"""
    df = read_sql(
        """
        SELECT p.trade_date, p.a_code, p.h_code, p.name, p.premium_pct,
               t.industry_flag, m.pair_end, m.exit_rule
        FROM trade_ah_premium_daily p
        JOIN trade_ah_tradable_daily t
          ON p.trade_date = t.trade_date
         AND p.a_code = t.a_code
         AND p.h_code = t.h_code
        LEFT JOIN trade_ah_pair_meta m
          ON p.a_code = m.a_code AND p.h_code = m.h_code
        WHERE t.is_tradable = 1
          AND p.trade_date >= %s
          AND p.premium_pct IS NOT NULL
        """,
        (START,),
    )
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['pair_end'] = pd.to_datetime(df['pair_end'])
    df['premium_pct'] = pd.to_numeric(df['premium_pct'], errors='coerce')
    df = df.dropna(subset=['premium_pct'])
    df['ym'] = df['trade_date'].dt.to_period('M')
    month_end = df.groupby('ym')['trade_date'].transform('max')
    sig = df[df['trade_date'] == month_end].copy()
    sig = sig.drop_duplicates(['trade_date', 'h_code'])
    print(f'月末信号 {sig.trade_date.nunique()} 个月, {len(sig)} 行, '
          f'{sig.h_code.nunique()} 只 H')
    return sig


def load_hfq(codes: list[str]) -> dict[str, pd.DataFrame]:
    frames = []
    conn = get_connection()
    try:
        for i in range(0, len(codes), 40):
            part = codes[i:i + 40]
            ph = ','.join(['%s'] * len(part))
            frames.append(pd.read_sql(
                f"""
                SELECT stock_code, trade_date, close_price
                FROM trade_stock_daily_hfq
                WHERE adjust_type='back'
                  AND stock_code IN ({ph})
                  AND trade_date >= %s
                  AND close_price > 0
                """,
                conn,
                params=part + [START],
            ))
    finally:
        conn.close()
    raw = pd.concat(frames, ignore_index=True)
    raw['trade_date'] = pd.to_datetime(raw['trade_date'])
    raw['close_price'] = pd.to_numeric(raw['close_price'], errors='coerce')
    out = {}
    for code, g in raw.dropna(subset=['close_price']).groupby('stock_code'):
        g = g.sort_values('trade_date').drop_duplicates('trade_date')
        out[code] = g.reset_index(drop=True)
    print(f'后复权行情 {len(out)} 只')
    return out


def forward_return(bars: pd.DataFrame, signal: pd.Timestamp, pair_end, n_hold: int):
    """下一交易日收盘买入，持有 n_hold 个交易日；窗口先结束则停在 pair_end。"""
    dates = bars['trade_date'].to_numpy()
    closes = bars['close_price'].to_numpy()
    entry_pos = int(np.searchsorted(dates, np.datetime64(signal), side='right'))
    if entry_pos >= len(dates) - 1:
        return np.nan, np.nan, 0
    entry_px = float(closes[entry_pos])
    if not np.isfinite(entry_px) or entry_px <= 0:
        return np.nan, np.nan, 0

    exit_pos = entry_pos + n_hold
    capped = 0
    if exit_pos >= len(dates):
        # 行情还没走到持有期末。只有窗口已经结束才允许用最后价，否则视为样本未完成。
        if pd.isna(pair_end) or dates[-1] < np.datetime64(pair_end):
            return np.nan, np.nan, 0
        exit_pos = len(dates) - 1
        capped = 1
    exit_date = pd.Timestamp(dates[exit_pos])
    if pd.notna(pair_end) and exit_date > pair_end:
        capped_pos = int(np.searchsorted(dates, np.datetime64(pair_end), side='right')) - 1
        if capped_pos <= entry_pos:
            return np.nan, np.nan, 0
        exit_pos = capped_pos
        capped = 1
    exit_px = float(closes[exit_pos])
    if not np.isfinite(exit_px) or exit_px <= 0:
        return np.nan, np.nan, 0
    return exit_px / entry_px - 1.0, pd.Timestamp(dates[exit_pos]), capped


def attach_forward(sig: pd.DataFrame, hfq: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for r in sig.itertuples(index=False):
        bars = hfq.get(r.h_code)
        item = {
            'trade_date': r.trade_date,
            'ym': str(r.ym),
            'a_code': r.a_code,
            'h_code': r.h_code,
            'name': r.name,
            'premium_pct': r.premium_pct,
            'industry_flag': r.industry_flag,
            'pair_end': r.pair_end,
            'exit_rule': r.exit_rule,
        }
        if bars is None:
            for label, _n in HORIZONS:
                item[f'ret_{label}'] = np.nan
                item[f'cap_{label}'] = 0
            rows.append(item)
            continue
        for label, n_hold in HORIZONS:
            ret, _exit_dt, capped = forward_return(bars, r.trade_date, r.pair_end, n_hold)
            item[f'ret_{label}'] = ret
            item[f'cap_{label}'] = capped
        rows.append(item)
    out = pd.DataFrame(rows)
    for label, _n in HORIZONS:
        n_ok = int(out[f'ret_{label}'].notna().sum())
        n_cap = int(((out[f'ret_{label}'].notna()) & (out[f'cap_{label}'] == 1)).sum())
        print(f'  {label} 有效收益 {n_ok}，其中窗口截断 {n_cap}')
    return out


def assign_quintile(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    if len(g) < MIN_NAMES:
        g['group'] = np.nan
        return g
    # 5 = 溢价最高 = H 最便宜
    g['group'] = pd.qcut(g['premium_pct'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    g['group'] = g['group'].astype(int)
    return g


def nw_mean_test(x: np.ndarray, lags: int) -> tuple[float, float, float, int]:
    """均值、Newey-West t、双侧 p、样本数。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, n
    mu = float(x.mean())
    e = x - mu
    gamma0 = float(np.dot(e, e) / n)
    var = gamma0
    lags = int(min(lags, n - 2))
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = float(np.dot(e[lag:], e[:-lag]) / n)
        var += 2.0 * w * gamma
    var = max(var, 0.0)
    se = np.sqrt(var / n) if var > 0 else np.nan
    if not np.isfinite(se) or se == 0:
        return mu, np.nan, np.nan, n
    t_val = mu / se
    p_val = float(2 * stats.t.sf(abs(t_val), df=n - 1))
    return mu, float(t_val), p_val, n


def monthly_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每月五组等权收益，以及当月截面 IC（溢价 vs 未来收益）。"""
    group_rows = []
    ic_rows = []
    for (ym, dt), g in panel.groupby(['ym', 'trade_date']):
        if g['group'].isna().all():
            continue
        rec = {
            'ym': ym,
            'trade_date': dt,
            'n': int(g['group'].notna().sum()),
            'premium_p50': float(g['premium_pct'].median()),
        }
        for label, _n in HORIZONS:
            col = f'ret_{label}'
            for k in range(1, 6):
                sub = g.loc[g['group'] == k, col]
                rec[f'{label}_g{k}'] = float(sub.mean()) if sub.notna().any() else np.nan
            hi = rec[f'{label}_g5']
            lo = rec[f'{label}_g1']
            rec[f'{label}_ls'] = hi - lo if np.isfinite(hi) and np.isfinite(lo) else np.nan
            use = g[['premium_pct', col]].dropna()
            if len(use) >= MIN_NAMES:
                ic = float(use['premium_pct'].corr(use[col], method='spearman'))
            else:
                ic = np.nan
            rec[f'ic_{label}'] = ic
        group_rows.append(rec)
        ic_rows.append({k: rec[k] for k in ('ym', 'trade_date', 'n', 'ic_1m', 'ic_3m', 'ic_6m')})
    return pd.DataFrame(group_rows), pd.DataFrame(ic_rows)


def summarize(monthly: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    for label, _n in HORIZONS:
        means = []
        for k in range(1, 6):
            mu, t_val, p_val, n = nw_mean_test(monthly[f'{label}_g{k}'].to_numpy(), NW_LAGS[label])
            means.append(mu)
            rows.append({
                'scope': scope, 'horizon': label, 'item': f'G{k}',
                'mean': mu, 't': t_val, 'p': p_val, 'n_months': n,
            })
        mu, t_val, p_val, n = nw_mean_test(monthly[f'{label}_ls'].to_numpy(), NW_LAGS[label])
        mono = all(np.isfinite(means)) and all(means[i] < means[i + 1] for i in range(4))
        ic_mu, ic_t, ic_p, ic_n = nw_mean_test(monthly[f'ic_{label}'].to_numpy(), NW_LAGS[label])
        ic_std = float(np.nanstd(monthly[f'ic_{label}'].to_numpy(), ddof=1))
        icir = ic_mu / ic_std if ic_std and np.isfinite(ic_std) and ic_std > 0 else np.nan
        rows.append({
            'scope': scope, 'horizon': label, 'item': 'G5-G1',
            'mean': mu, 't': t_val, 'p': p_val, 'n_months': n,
            'monotonic': int(mono), 'ic_mean': ic_mu, 'ic_t': ic_t, 'ic_p': ic_p,
            'icir': icir, 'ic_months': ic_n,
        })
    return pd.DataFrame(rows)


def plot_result(monthly: pd.DataFrame, summary: pd.DataFrame, scope: str, title: str, filename: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    x = np.arange(1, 6)
    width = 0.25
    part_scope = summary[summary.scope == scope]
    for i, (label, _n) in enumerate(HORIZONS):
        part = part_scope[(part_scope.horizon == label) & (part_scope.item.str.fullmatch(r'G[1-5]'))]
        vals = [float(part.loc[part.item == f'G{k}', 'mean'].iloc[0]) * 100 for k in range(1, 6)]
        ax.bar(x + (i - 1) * width, vals, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(['G1 溢价最低', 'G2', 'G3', 'G4', 'G5 溢价最高'])
    ax.set_ylabel('组内等权平均收益 (%)')
    ax.set_title(f'{title}：未来 H 后复权收益')
    ax.axhline(0, color='#888', lw=0.8)
    ax.legend()

    ax = axes[1]
    nav = (1 + monthly['1m_ls'].fillna(0)).cumprod()
    ax.plot(monthly['trade_date'], nav, color='#6aa4ff')
    ax.set_title(f'{title} G5-G1 月频多空净值（持有 1 个月，未扣成本）')
    ax.set_ylabel('净值')
    ax.axhline(1, color='#888', lw=0.8)
    fig.tight_layout()
    path = OUT / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'图 -> {path}')


def _fmt(v, pct=False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return ''
    if pct:
        return f'{float(v) * 100:.2f}%'
    return f'{float(v):.2f}'


def _table_block(summary: pd.DataFrame, scope: str) -> list[str]:
    part_scope = summary[summary.scope == scope]
    lines = ['<table><tr><th>期限</th><th>G1</th><th>G2</th><th>G3</th><th>G4</th><th>G5</th>'
             '<th>G5-G1</th><th>t</th><th>p</th><th>单调</th><th>IC</th><th>ICIR</th><th>月数</th></tr>']
    for label, _n in HORIZONS:
        part = part_scope[part_scope.horizon == label]
        cells = [_fmt(float(part.loc[part.item == f'G{k}', 'mean'].iloc[0]), pct=True) for k in range(1, 6)]
        ls = part[part.item == 'G5-G1'].iloc[0]
        mono = '是' if int(ls['monotonic']) == 1 else '否'
        p_txt = '' if pd.isna(ls['p']) else f"{float(ls['p']):.4f}"
        lines.append(
            f"<tr><td>{label}</td>" + ''.join(f'<td>{c}</td>' for c in cells)
            + f"<td>{_fmt(ls['mean'], pct=True)}</td><td>{_fmt(ls['t'])}</td><td>{p_txt}</td>"
            + f"<td>{mono}</td><td>{_fmt(ls['ic_mean'])}</td><td>{_fmt(ls['icir'])}</td>"
            + f"<td>{int(ls['n_months'])}</td></tr>"
        )
    lines.append('</table>')
    return lines


def write_html(summary: pd.DataFrame, monthly_map: dict[str, pd.DataFrame]):
    sections = (
        ('all', '全样本', '02_纯折价五分组.png', '全池一起五等分'),
        ('finance', '金融', '02_纯折价五分组_金融.png', '只在金融股内部五等分'),
        ('non_finance', '非金融', '02_纯折价五分组_非金融.png', '只在非金融股内部五等分'),
    )
    lines = []
    lines.append('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">')
    lines.append('<title>纯折价五分组</title>')
    lines.append('<style>body{font-family:"Microsoft YaHei",sans-serif;background:#181818;color:#ececec;padding:28px;}'
                 'h2{margin-top:36px;}table{border-collapse:collapse;margin:12px 0 24px;}'
                 'td,th{border-bottom:1px solid #3d3d3d;padding:6px 10px;text-align:right;}'
                 'th{color:#a8a8a8;}td:first-child,th:first-child{text-align:left;}'
                 'img{max-width:960px;background:#111;}</style></head><body>')
    lines.append('<h1>纯折价五分组</h1>')
    lines.append('<p>规划 1.0 第 2 步。样本 2017-01-03 起，月末可新开仓，按不复权溢价五等分。'
                 'G5 为溢价最高（H 最便宜）。收益为 H 后复权，下一交易日收盘买入。'
                 '金融 / 非金融是在各自池子里重新分组，不是从全样本分组里再拆。</p>')
    for scope, title, img, note in sections:
        monthly = monthly_map[scope]
        lines.append(f'<h2>{title}</h2>')
        lines.append(f'<p>{note}。信号月数 {len(monthly)}，'
                     f'最后信号日 {pd.Timestamp(monthly["trade_date"].max()).date()}。</p>')
        lines.append(f'<img src="{img}" alt="{title}">')
        lines.extend(_table_block(summary, scope))
    lines.append('<p>t / p 为月度序列均值的 Newey-West 检验。1 个月滞后 0，3 个月滞后 2，6 个月滞后 5。未扣交易成本。</p>')
    lines.append('</body></html>')
    path = OUT / '02_纯折价五分组.html'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'结果页 -> {path}')


def main():
    if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(parents=True, exist_ok=True)
    print('读取月末可交易溢价...')
    sig = load_signals()
    print('读取 H 后复权...')
    hfq = load_hfq(sorted(sig['h_code'].unique()))
    print('计算未来收益...')
    panel = attach_forward(sig, hfq)
    parts = []
    for _, g in panel.groupby('trade_date'):
        parts.append(assign_quintile(g))
    panel = pd.concat(parts, ignore_index=True)
    panel.to_csv(OUT / '02_月末信号与未来收益.csv', index=False, encoding='utf-8-sig')

    monthly, ic = monthly_panel(panel.dropna(subset=['group']))
    monthly.to_csv(OUT / '02_分组月收益.csv', index=False, encoding='utf-8-sig')
    ic.to_csv(OUT / '02_月度IC.csv', index=False, encoding='utf-8-sig')

    monthly_map = {'all': monthly}
    summaries = [summarize(monthly, 'all')]
    scope_file = {'finance': '02_分组月收益_金融.csv', 'non_finance': '02_分组月收益_非金融.csv'}
    for flag, scope in (('finance', 'finance'), ('non_finance', 'non_finance')):
        sub = panel[panel['industry_flag'] == flag]
        sub_parts = [assign_quintile(g) for _, g in sub.groupby('trade_date')]
        sub_panel = pd.concat(sub_parts, ignore_index=True) if sub_parts else pd.DataFrame()
        sub_m, sub_ic = monthly_panel(sub_panel.dropna(subset=['group'])) if not sub_panel.empty else (pd.DataFrame(), pd.DataFrame())
        if sub_m.empty:
            raise RuntimeError(f'{scope} 没有足够的月末样本')
        sub_m.to_csv(OUT / scope_file[scope], index=False, encoding='utf-8-sig')
        sub_ic.to_csv(OUT / scope_file[scope].replace('分组月收益', '月度IC'), index=False, encoding='utf-8-sig')
        monthly_map[scope] = sub_m
        summaries.append(summarize(sub_m, scope))
        print(f'{scope} 信号月 {len(sub_m)}，月均只数 {sub_m["n"].mean():.1f}')
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUT / '02_汇总.csv', index=False, encoding='utf-8-sig')
    plot_result(monthly_map['all'], summary, 'all', '全样本', '02_纯折价五分组.png')
    plot_result(monthly_map['finance'], summary, 'finance', '金融', '02_纯折价五分组_金融.png')
    plot_result(monthly_map['non_finance'], summary, 'non_finance', '非金融', '02_纯折价五分组_非金融.png')
    write_html(summary, monthly_map)

    show = summary[summary.item == 'G5-G1'][
        ['scope', 'horizon', 'mean', 't', 'p', 'monotonic', 'ic_mean', 'icir', 'n_months']
    ]
    print('\nG5-G1:')
    print(show.to_string(index=False))
    print('完成')


if __name__ == '__main__':
    main()
