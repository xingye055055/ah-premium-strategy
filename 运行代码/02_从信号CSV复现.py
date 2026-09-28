# -*- coding: utf-8 -*-
# Author：@跟陈博士学AI
"""
用已经导出的月末信号 CSV 复现全样本、金融、非金融五分组。

不连接数据库。读上级目录「原始数据/02_月末信号与未来收益.csv」，
按当时的规则重新分组，重写汇总、图和结果页。
不覆盖这份信号 CSV。

运行：
  python 02_从信号CSV复现.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ah_q', ROOT / '02_纯折价五分组.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def main():
    if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    src = q.DATA / '02_月末信号与未来收益.csv'
    if not src.exists():
        raise FileNotFoundError(f'缺少信号表 {src}')
    panel = q.pd.read_csv(src, parse_dates=['trade_date'])
    panel['premium_pct'] = q.pd.to_numeric(panel['premium_pct'], errors='coerce')
    for label, _n in q.HORIZONS:
        panel[f'ret_{label}'] = q.pd.to_numeric(panel[f'ret_{label}'], errors='coerce')
    panel = panel.drop(columns=['group'], errors='ignore')
    print(f'读入信号 {panel.trade_date.nunique()} 个月, {len(panel)} 行, {panel.h_code.nunique()} 只 H')

    parts = [q.assign_quintile(g) for _, g in panel.groupby('trade_date')]
    panel = q.pd.concat(parts, ignore_index=True)
    monthly, ic = q.monthly_panel(panel.dropna(subset=['group']))
    monthly.to_csv(q.OUT / '02_分组月收益.csv', index=False, encoding='utf-8-sig')
    ic.to_csv(q.OUT / '02_月度IC.csv', index=False, encoding='utf-8-sig')

    monthly_map = {'all': monthly}
    summaries = [q.summarize(monthly, 'all')]
    scope_file = {'finance': '02_分组月收益_金融.csv', 'non_finance': '02_分组月收益_非金融.csv'}
    for flag, scope in (('finance', 'finance'), ('non_finance', 'non_finance')):
        sub = panel[panel['industry_flag'] == flag]
        sub_parts = [q.assign_quintile(g) for _, g in sub.groupby('trade_date')]
        sub_panel = q.pd.concat(sub_parts, ignore_index=True)
        sub_m, sub_ic = q.monthly_panel(sub_panel.dropna(subset=['group']))
        sub_m.to_csv(q.OUT / scope_file[scope], index=False, encoding='utf-8-sig')
        sub_ic.to_csv(q.OUT / scope_file[scope].replace('分组月收益', '月度IC'), index=False, encoding='utf-8-sig')
        monthly_map[scope] = sub_m
        summaries.append(q.summarize(sub_m, scope))
        print(f'{scope} 信号月 {len(sub_m)}，月均只数 {sub_m["n"].mean():.1f}')
    summary = q.pd.concat(summaries, ignore_index=True)
    summary.to_csv(q.OUT / '02_汇总.csv', index=False, encoding='utf-8-sig')
    q.plot_result(monthly_map['all'], summary, 'all', '全样本', '02_纯折价五分组.png')
    q.plot_result(monthly_map['finance'], summary, 'finance', '金融', '02_纯折价五分组_金融.png')
    q.plot_result(monthly_map['non_finance'], summary, 'non_finance', '非金融', '02_纯折价五分组_非金融.png')
    q.write_html(summary, monthly_map)
    show = summary[summary.item == 'G5-G1'][
        ['scope', 'horizon', 'mean', 't', 'p', 'monotonic', 'ic_mean', 'icir', 'n_months']
    ]
    print('\nG5-G1:')
    print(show.to_string(index=False))
    print('完成')


if __name__ == '__main__':
    main()
