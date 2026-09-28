# -*- coding: utf-8 -*-
# Author：@跟陈博士学AI
"""
最高溢价组相对全板块等权的超额：上涨月、下跌月，以及连续上涨超过 3 个月的阶段。

全板块等权 = 当月该板块里有 1 个月收益的全部股票简单平均，不是五组再平均。
上涨月：这个等权收益 > 0。下跌月：<= 0。用的是同一个持有月，属于事后描述。
连续上涨：信号月上板块等权收益连续为正，且长度超过 3 个月（至少 4 个月）。

不连接数据库。读上级目录「原始数据」里的月末信号和行业大类。
图、表、HTML 写到「课题研究结论产出」。

运行：
  python 04_周期超额.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('s3', ROOT / '03_分板块五分组.py')
s3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s3)
q = s3.q

OUT = s3.OUT
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def month_rows(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """每个板块、每个信号月：最高溢价组收益、全板块等权、超额。"""
    rows = []
    for name in names:
        sub = df[df['industry'] == name]
        n_groups = s3.choose_n_groups(sub)
        parts = [s3.assign_groups(g, n_groups) for _, g in sub.groupby('trade_date')]
        panel = pd.concat(parts, ignore_index=True)
        for dt, g in panel.groupby('trade_date'):
            g = g.dropna(subset=['group', 'ret_1m'])
            if len(g) < n_groups:
                continue
            hi = g.loc[g['group'] == n_groups, 'ret_1m']
            if hi.empty:
                continue
            sec = float(g['ret_1m'].mean())
            ghi = float(hi.mean())
            rows.append({
                '板块': name,
                'trade_date': dt,
                '组数': n_groups,
                '只数': int(len(g)),
                '最高组只数': int(len(hi)),
                '板块等权': sec,
                '最高组': ghi,
                '超额': ghi - sec,
            })
    return pd.DataFrame(rows)


def pack(name: str, n_groups: int, part: pd.DataFrame, regime: str) -> dict:
    mu, t_val, p_val, n = q.nw_mean_test(part['超额'].to_numpy(), 0)
    return {
        '板块': name,
        '组数': n_groups,
        '周期': regime,
        '月数': n,
        '超额': mu,
        't': t_val,
        'p': p_val,
        '最高组收益': float(part['最高组'].mean()) if n else np.nan,
        '板块等权': float(part['板块等权'].mean()) if n else np.nan,
        '超额为正占比': float((part['超额'] > 0).mean()) if n else np.nan,
    }


def up_streaks(g: pd.DataFrame, min_len: int) -> list[pd.DataFrame]:
    """信号月顺序上，板块等权连续为正、且长度达到 min_len 的段。"""
    g = g.sort_values('trade_date')
    up = g['板块等权'].to_numpy() > 0
    if len(up) == 0:
        return []
    run = np.cumsum(np.r_[True, up[1:] != up[:-1]])
    kept = []
    tagged = g.copy()
    tagged['run'] = run
    tagged['up'] = up
    for _, part in tagged.groupby('run'):
        if bool(part['up'].iloc[0]) and len(part) >= min_len:
            kept.append(part.drop(columns=['run', 'up']))
    return kept


def phase_table(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    detail_rows = []
    for name, g in monthly.groupby('板块', sort=False):
        n_groups = int(g['组数'].iloc[0])
        runs = up_streaks(g, 4)
        months = pd.concat(runs, ignore_index=True) if runs else g.iloc[0:0]
        mu, t_val, p_val, n = q.nw_mean_test(months['超额'].to_numpy() if len(months) else [], 0)
        phase_ex = []
        for i, part in enumerate(runs, 1):
            sec = float(np.prod(1 + part['板块等权'].to_numpy()) - 1)
            hi = float(np.prod(1 + part['最高组'].to_numpy()) - 1)
            phase_ex.append(hi - sec)
            detail_rows.append({
                '板块': name,
                '阶段': i,
                '开始': pd.Timestamp(part['trade_date'].iloc[0]).date().isoformat(),
                '结束': pd.Timestamp(part['trade_date'].iloc[-1]).date().isoformat(),
                '月数': len(part),
                '板块累计': sec,
                '最高组累计': hi,
                '累计超额': hi - sec,
                '月均超额': float(part['超额'].mean()),
            })
        summary_rows.append({
            '板块': name,
            '组数': n_groups,
            '阶段数': len(runs),
            '阶段内月数': int(n) if n else 0,
            '月均超额': mu,
            '月t': t_val,
            '月p': p_val,
            '月超额为正占比': float((months['超额'] > 0).mean()) if len(months) else np.nan,
            '阶段累计超额为正': int(np.sum(np.array(phase_ex) > 0)) if phase_ex else 0,
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def plot_up_down(summary: pd.DataFrame):
    show = summary[summary['周期'].isin(['上涨', '下跌'])].copy()
    order = show[show['周期'] == '上涨'].sort_values('超额', ascending=False)['板块'].tolist()
    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = np.arange(len(order))
    width = 0.38
    upv = [float(show[(show['板块'] == n) & (show['周期'] == '上涨')]['超额'].iloc[0]) * 100 for n in order]
    dnv = [float(show[(show['板块'] == n) & (show['周期'] == '下跌')]['超额'].iloc[0]) * 100 for n in order]
    ax.bar(x - width / 2, upv, width=width, label='上涨月', color='#d4553a')
    ax.bar(x + width / 2, dnv, width=width, label='下跌月', color='#4c78a8')
    ax.axhline(0, color='#888', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha='right')
    ax.set_ylabel('最高溢价组减板块等权 (%)')
    ax.set_title('最高溢价组相对全板块等权的月超额（持有 1 个月，未扣成本）')
    ax.legend()
    fig.tight_layout()
    path = OUT / '04_最高组相对板块超额.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'图 -> {path}')


def _cell(v, pct=False, digits=2) -> str:
    if v is None or pd.isna(v):
        return ''
    if pct:
        return f'{float(v) * 100:.2f}%'
    return f'{float(v):.{digits}f}'


def write_html(summary: pd.DataFrame, phase_sum: pd.DataFrame, phase_detail: pd.DataFrame):
    """上涨月、下跌月和连续上涨阶段的 HTML 报告。"""
    lines = [
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">',
        '<title>周期里的超额</title>',
        '<style>body{font-family:"Microsoft YaHei",sans-serif;background:#181818;color:#ececec;padding:28px;}'
        'h2{margin-top:36px;}table{border-collapse:collapse;margin:12px 0 24px;}'
        'td,th{border-bottom:1px solid #3d3d3d;padding:6px 10px;text-align:right;}'
        'th{color:#a8a8a8;}td:first-child,th:first-child{text-align:left;}'
        'img{max-width:960px;background:#111;}p{line-height:1.6;max-width:880px;}</style></head><body>',
        '<h1>最高溢价组相对全板块等权</h1>',
        '<p>全板块等权是当月该板块里有 1 个月收益的全部股票简单平均。'
        '上涨月指这个等权收益大于 0，下跌月指小于等于 0。'
        '连续上涨指信号月上这个等权连续为正，并且这段长于 3 个月。'
        '持有 1 个月，未扣成本。四组板块的最高组是第 4 组。</p>',
        '<p>这个切分用的是持有期里板块自己的涨跌，用来描述已经发生的行情，不是月末已经知道的下单条件。</p>',
        '<img src="04_最高组相对板块超额.png" alt="上涨月与下跌月超额">',
        '<h2>上涨月与下跌月</h2>',
        '<table><tr><th>板块</th><th>组数</th><th>周期</th><th>月数</th>'
        '<th>超额</th><th>t</th><th>p</th><th>超额为正占比</th></tr>',
    ]
    show = summary[summary['周期'].isin(['上涨', '下跌'])]
    for r in show.itertuples(index=False):
        lines.append(
            f'<tr><td>{r.板块}</td><td>{int(r.组数)}</td><td>{r.周期}</td><td>{int(r.月数)}</td>'
            f'<td>{_cell(r.超额, pct=True)}</td><td>{_cell(r.t)}</td><td>{_cell(r.p, digits=4)}</td>'
            f'<td>{_cell(r.超额为正占比, pct=True)}</td></tr>'
        )
    lines.append('</table>')
    lines.append('<h2>连续上涨超过 3 个月</h2>')
    lines.append(
        '<table><tr><th>板块</th><th>阶段数</th><th>阶段内月数</th><th>月均超额</th>'
        '<th>t</th><th>p</th><th>月超额为正占比</th><th>累计超额为正的段数</th></tr>'
    )
    for r in phase_sum.itertuples(index=False):
        lines.append(
            f'<tr><td>{r.板块}</td><td>{int(r.阶段数)}</td><td>{int(r.阶段内月数)}</td>'
            f'<td>{_cell(r.月均超额, pct=True)}</td><td>{_cell(r.月t)}</td><td>{_cell(r.月p, digits=4)}</td>'
            f'<td>{_cell(r.月超额为正占比, pct=True)}</td><td>{int(r.阶段累计超额为正)}</td></tr>'
        )
    lines.append('</table>')
    if len(phase_detail):
        pos = int((phase_detail['累计超额'] > 0).sum())
        lines.append(f'<p>全部阶段 {len(phase_detail)} 段，最高溢价组累计超过板块的有 {pos} 段。各段起止见同目录 CSV。</p>')
    lines.append('<p>t / p 与第 2 步相同，是月度超额均值的检验。1 个月滞后取 0。p 低于 0.05 记为显著。</p>')
    lines.append('</body></html>')
    path = OUT / '04_周期超额.html'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'结果页 -> {path}')


def main():
    if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    df = s3.load_panel()
    names = s3.sectors_over(df)
    monthly = month_rows(df, names)
    monthly.to_csv(OUT / '04_最高组相对板块超额_月度.csv', index=False, encoding='utf-8-sig')

    rows = []
    for name, g in monthly.groupby('板块', sort=False):
        n_groups = int(g['组数'].iloc[0])
        rows.append(pack(name, n_groups, g[g['板块等权'] > 0], '上涨'))
        rows.append(pack(name, n_groups, g[g['板块等权'] <= 0], '下跌'))
        rows.append(pack(name, n_groups, g, '全部'))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / '04_最高组相对板块超额.csv', index=False, encoding='utf-8-sig')
    plot_up_down(summary)

    phase_sum, phase_detail = phase_table(monthly)
    phase_sum.to_csv(OUT / '04_持续上涨阶段超额.csv', index=False, encoding='utf-8-sig')
    phase_detail.to_csv(OUT / '04_持续上涨阶段明细.csv', index=False, encoding='utf-8-sig')
    write_html(summary, phase_sum, phase_detail)

    view = summary[summary['周期'].isin(['上涨', '下跌'])][
        ['板块', '组数', '周期', '月数', '超额', 't', 'p', '超额为正占比']
    ]
    print('\n上涨月 / 下跌月 最高组减板块等权:')
    print(view.to_string(index=False))
    print('\n连续上涨超过 3 个月:')
    print(phase_sum.to_string(index=False))
    if len(phase_detail):
        pos = int((phase_detail['累计超额'] > 0).sum())
        print(f'全部阶段 {len(phase_detail)}，最高组累计超过板块 {pos}')
    print('完成')


if __name__ == '__main__':
    main()
