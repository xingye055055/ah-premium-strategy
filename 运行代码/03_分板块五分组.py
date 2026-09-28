# -*- coding: utf-8 -*-
# Author：@跟陈博士学AI
"""
超过 4 只的申万一级，各自在板块内部按溢价分组。

能稳定做出五组的板块用五组（至少 36 个月、当月不少于 5 只）。
五组月份不够的板块整段改成四组（当月不少于 4 只）。
收益沿用 02_月末信号与未来收益.csv，不再重算行情。

运行：
  python 03_分板块五分组.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PKG = ROOT.parent
DATA = PKG / '原始数据'
OUT = PKG / '课题研究结论产出'
MIN_SECTOR = 4
MIN_MONTHS_FOR_5 = 36

spec = importlib.util.spec_from_file_location('ah_q', ROOT / '02_纯折价五分组.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
def load_panel() -> pd.DataFrame:
    sig = pd.read_csv(DATA / '02_月末信号与未来收益.csv', parse_dates=['trade_date'])
    ind = pd.read_csv(DATA / 'AH交集行业大类.csv', dtype=str)
    ind = ind.rename(columns={'H代码': 'h_code', '行业大类': 'industry', '名称': 'ind_name'})
    ind = ind[['h_code', 'industry', 'ind_name']].drop_duplicates('h_code')
    df = sig.merge(ind, on='h_code', how='left')
    miss = int(df['industry'].isna().sum())
    if miss:
        print(f'信号里有 {miss} 行对不上行业，已丢掉')
        df = df.dropna(subset=['industry'])
    return df


def sectors_over(df: pd.DataFrame) -> list[str]:
    """按行业名单里的股票只数，保留超过 4 只的板块。"""
    n = df.drop_duplicates('h_code').groupby('industry')['h_code'].nunique()
    keep = n[n > MIN_SECTOR].sort_values(ascending=False)
    print('超过 4 只的板块:')
    for name, cnt in keep.items():
        print(f'  {name} {int(cnt)}')
    return list(keep.index)


def choose_n_groups(sub: pd.DataFrame) -> int:
    """五组月份够 36 个就用五组，否则整段改四组。"""
    sizes = sub.groupby('trade_date').size()
    n5 = int((sizes >= 5).sum())
    return 5 if n5 >= MIN_MONTHS_FOR_5 else 4


def assign_groups(g: pd.DataFrame, n_groups: int) -> pd.DataFrame:
    g = g.copy()
    if len(g) < n_groups:
        g['group'] = np.nan
        return g
    g['group'] = pd.qcut(
        g['premium_pct'].rank(method='first'), n_groups, labels=list(range(1, n_groups + 1))
    ).astype(int)
    return g


def monthly_panel(panel: pd.DataFrame, n_groups: int) -> pd.DataFrame:
    rows = []
    for (ym, dt), g in panel.groupby(['ym', 'trade_date']):
        if g['group'].isna().all():
            continue
        rec = {
            'ym': ym,
            'trade_date': dt,
            'n': int(g['group'].notna().sum()),
            'n_groups': n_groups,
        }
        for label, _n in q.HORIZONS:
            col = f'ret_{label}'
            for k in range(1, n_groups + 1):
                sub = g.loc[g['group'] == k, col]
                rec[f'{label}_g{k}'] = float(sub.mean()) if sub.notna().any() else np.nan
            hi = rec[f'{label}_g{n_groups}']
            lo = rec[f'{label}_g1']
            rec[f'{label}_ls'] = hi - lo if np.isfinite(hi) and np.isfinite(lo) else np.nan
            use = g[['premium_pct', col]].dropna()
            if len(use) >= n_groups:
                rec[f'ic_{label}'] = float(use['premium_pct'].corr(use[col], method='spearman'))
            else:
                rec[f'ic_{label}'] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(monthly: pd.DataFrame, scope: str, n_groups: int) -> pd.DataFrame:
    rows = []
    hi_name = f'G{n_groups}-G1'
    for label, _n in q.HORIZONS:
        means = []
        for k in range(1, n_groups + 1):
            mu, t_val, p_val, n = q.nw_mean_test(monthly[f'{label}_g{k}'].to_numpy(), q.NW_LAGS[label])
            means.append(mu)
            rows.append({
                'scope': scope, 'n_groups': n_groups, 'horizon': label, 'item': f'G{k}',
                'mean': mu, 't': t_val, 'p': p_val, 'n_months': n,
            })
        mu, t_val, p_val, n = q.nw_mean_test(monthly[f'{label}_ls'].to_numpy(), q.NW_LAGS[label])
        mono = all(np.isfinite(means)) and all(means[i] < means[i + 1] for i in range(len(means) - 1))
        ic_mu, ic_t, ic_p, ic_n = q.nw_mean_test(monthly[f'ic_{label}'].to_numpy(), q.NW_LAGS[label])
        ic_std = float(np.nanstd(monthly[f'ic_{label}'].to_numpy(), ddof=1))
        icir = ic_mu / ic_std if ic_std and np.isfinite(ic_std) and ic_std > 0 else np.nan
        rows.append({
            'scope': scope, 'n_groups': n_groups, 'horizon': label, 'item': hi_name,
            'mean': mu, 't': t_val, 'p': p_val, 'n_months': n,
            'monotonic': int(mono), 'ic_mean': ic_mu, 'ic_t': ic_t, 'ic_p': ic_p,
            'icir': icir, 'ic_months': ic_n,
        })
    return pd.DataFrame(rows)


def one_sector(df: pd.DataFrame, name: str) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    sub = df[df['industry'] == name]
    n_groups = choose_n_groups(sub)
    parts = [assign_groups(g, n_groups) for _, g in sub.groupby('trade_date')]
    panel = pd.concat(parts, ignore_index=True)
    monthly = monthly_panel(panel.dropna(subset=['group']), n_groups)
    if monthly.empty:
        return n_groups, monthly, pd.DataFrame()
    return n_groups, monthly, summarize(monthly, name, n_groups)


def plot_sector(monthly: pd.DataFrame, summary: pd.DataFrame, name: str, n_groups: int, n_names: int):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    x = np.arange(1, n_groups + 1)
    width = 0.25
    part_scope = summary[summary.scope == name]
    for i, (label, _n) in enumerate(q.HORIZONS):
        part = part_scope[(part_scope.horizon == label) & (part_scope.item.str.fullmatch(r'G[1-9]'))]
        vals = [float(part.loc[part.item == f'G{k}', 'mean'].iloc[0]) * 100 for k in range(1, n_groups + 1)]
        ax.bar(x + (i - 1) * width, vals, width=width, label=label)
    labels = [f'G{k}' for k in range(1, n_groups + 1)]
    labels[0] = 'G1 溢价最低'
    labels[-1] = f'G{n_groups} 溢价最高'
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('组内等权平均收益 (%)')
    ax.set_title(f'{name}（{n_names}只，{n_groups}组）：未来 H 后复权收益')
    ax.axhline(0, color='#888', lw=0.8)
    ax.legend()

    ax = axes[1]
    nav = (1 + monthly['1m_ls'].fillna(0)).cumprod()
    ax.plot(monthly['trade_date'], nav, color='#6aa4ff')
    ax.set_title(f'{name} G{n_groups}-G1 月频多空净值（持有 1 个月，未扣成本）')
    ax.set_ylabel('净值')
    ax.axhline(1, color='#888', lw=0.8)
    fig.tight_layout()
    path = OUT / f'03_五分组_{name}.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'图 -> {path}')


def table_block(summary: pd.DataFrame, name: str, n_groups: int) -> list[str]:
    part_scope = summary[summary.scope == name]
    hi = f'G{n_groups}-G1'
    heads = ''.join(f'<th>G{k}</th>' for k in range(1, n_groups + 1))
    lines = [f'<table><tr><th>期限</th>{heads}<th>{hi}</th><th>t</th><th>p</th>'
             '<th>单调</th><th>IC</th><th>ICIR</th><th>月数</th></tr>']
    for label, _n in q.HORIZONS:
        part = part_scope[part_scope.horizon == label]
        cells = ''.join(
            f'<td>{q._fmt(float(part.loc[part.item == f"G{k}", "mean"].iloc[0]), pct=True)}</td>'
            for k in range(1, n_groups + 1)
        )
        ls = part[part.item == hi].iloc[0]
        mono = '是' if int(ls['monotonic']) == 1 else '否'
        p_txt = '' if pd.isna(ls['p']) else f'{float(ls["p"]):.4f}'
        lines.append(
            f'<tr><td>{label}</td>{cells}<td>{q._fmt(ls["mean"], pct=True)}</td>'
            f'<td>{q._fmt(ls["t"])}</td><td>{p_txt}</td><td>{mono}</td>'
            f'<td>{q._fmt(ls["ic_mean"])}</td><td>{q._fmt(ls["icir"])}</td>'
            f'<td>{int(ls["n_months"])}</td></tr>'
        )
    lines.append('</table>')
    return lines


def write_html(summary, monthly_map, counts, group_of):
    lines = []
    lines.append('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">')
    lines.append('<title>分板块纯折价分组</title>')
    lines.append(
        '<style>body{font-family:"Microsoft YaHei",sans-serif;background:#181818;color:#ececec;padding:28px;}'
        'h2{margin-top:36px;}table{border-collapse:collapse;margin:12px 0 24px;}'
        'td,th{border-bottom:1px solid #3d3d3d;padding:6px 10px;text-align:right;}'
        'th{color:#a8a8a8;}td:first-child,th:first-child{text-align:left;}'
        'img{max-width:960px;background:#111;}</style></head><body>'
    )
    lines.append('<h1>分板块纯折价分组</h1>')
    lines.append(
        '<p>只做行业名单里超过 4 只的申万一级，在板块内部按不复权溢价分组。'
        '最高组是溢价最高（H 最便宜）。五组能凑满 36 个月的板块用五组；'
        '凑不满的整段改成四组。收益口径与全样本相同：2017-01-03 起，H 后复权，'
        '下一交易日收盘买入。未扣成本。</p>'
    )
    lines.append('<h2>1 个月最高组减最低组</h2>')
    lines.append('<table><tr><th>板块</th><th>分组</th><th>名单只数</th><th>信号月</th><th>月均只数</th>'
                 '<th>最高-最低</th><th>t</th><th>p</th><th>单调</th><th>IC</th></tr>')
    for name, monthly in monthly_map.items():
        n_groups = group_of[name]
        hi = f'G{n_groups}-G1'
        part = summary[(summary.scope == name) & (summary.horizon == '1m') & (summary.item == hi)]
        if part.empty:
            continue
        ls = part.iloc[0]
        mono = '是' if int(ls['monotonic']) == 1 else '否'
        p_txt = '' if pd.isna(ls['p']) else f'{float(ls["p"]):.4f}'
        lines.append(
            f'<tr><td>{name}</td><td>{n_groups}组</td><td>{counts[name]}</td>'
            f'<td>{len(monthly)}</td><td>{monthly["n"].mean():.1f}</td>'
            f'<td>{q._fmt(ls["mean"], pct=True)}</td><td>{q._fmt(ls["t"])}</td>'
            f'<td>{p_txt}</td><td>{mono}</td><td>{q._fmt(ls["ic_mean"])}</td></tr>'
        )
    lines.append('</table>')
    for name, monthly in monthly_map.items():
        n_groups = group_of[name]
        lines.append(f'<h2>{name}</h2>')
        lines.append(
            f'<p>{n_groups} 组。名单 {counts[name]} 只，信号月 {len(monthly)}，'
            f'月均可分组 {monthly["n"].mean():.1f} 只，'
            f'最后信号日 {pd.Timestamp(monthly["trade_date"].max()).date()}。</p>'
        )
        lines.append(f'<img src="03_五分组_{name}.png" alt="{name}">')
        lines.extend(table_block(summary, name, n_groups))
    lines.append('<p>t / p 为月度序列均值的 Newey-West 检验。1 个月滞后 0，3 个月滞后 2，6 个月滞后 5。</p>')
    lines.append('</body></html>')
    path = OUT / '03_分板块五分组.html'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'结果页 -> {path}')


def main():
    if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    df = load_panel()
    names = sectors_over(df)
    counts = df.drop_duplicates('h_code').groupby('industry')['h_code'].nunique().to_dict()
    monthly_map = {}
    group_of = {}
    summaries = []
    for name in names:
        n_groups, monthly, summary = one_sector(df, name)
        if monthly.empty:
            print(f'{name} 没有月份凑满 {n_groups} 只，跳过')
            continue
        monthly_map[name] = monthly
        group_of[name] = n_groups
        summaries.append(summary)
        monthly.to_csv(OUT / f'03_分组月收益_{name}.csv', index=False, encoding='utf-8-sig')
        print(f'{name} {n_groups}组，信号月 {len(monthly)}，月均 {monthly["n"].mean():.1f}')
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUT / '03_分板块汇总.csv', index=False, encoding='utf-8-sig')
    for name, monthly in monthly_map.items():
        plot_sector(monthly, summary, name, group_of[name], int(counts[name]))
    write_html(summary, monthly_map, counts, group_of)
    show = summary[(summary.horizon == '1m') & (summary.item.str.contains('-G1'))][
        ['scope', 'n_groups', 'item', 'mean', 't', 'p', 'monotonic', 'ic_mean', 'n_months']
    ]
    print('\n1个月 最高组-最低组:')
    print(show.to_string(index=False))
    print('完成')


if __name__ == '__main__':
    main()
