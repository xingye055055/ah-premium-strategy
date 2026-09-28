# -*- coding: utf-8 -*-
# Author：@跟陈博士学AI
"""
一次跑完三份报告。

读「原始数据」，把图、表和 HTML 写到「课题研究结论产出」。
不连接数据库。

运行：
  python 运行全部.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / '课题研究结论产出'
STEPS = ('02_从信号CSV复现.py', '03_分板块五分组.py', '04_周期超额.py')


def write_index():
    """入口页，链到三份已经写好的 HTML。"""
    lines = [
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">',
        '<title>AH股折溢价策略研究报告</title>',
        '<style>body{font-family:"Microsoft YaHei",sans-serif;background:#181818;color:#ececec;padding:28px;}'
        'a{color:#8eb6ff;}li{margin:8px 0;}p{line-height:1.7;max-width:820px;}</style></head><body>',
        '<h1>AH股折溢价策略研究报告</h1>',
        '<p>下面三份页面是本次运行刚写出来的。数字以页面里的表为准。</p>',
        '<ol>',
        '<li><a href="02_纯折价五分组.html">纯折价五分组</a>：全样本、金融、非金融。问溢价越高，未来 H 是否越好。</li>',
        '<li><a href="03_分板块五分组.html">分板块分组</a>：超过 4 只的行业，在板块内部再分组。</li>',
        '<li><a href="04_周期超额.html">上涨与下跌周期里的超额</a>：最高溢价组相对全板块等权，以及连续上涨超过 3 个月的阶段。</li>',
        '</ol>',
        '<p>第 2 步的通过标准是高溢价组显著优于低溢价组，或五组至少单调。这一步没有通过，所以没有继续做持仓和叠估值。</p>',
        '</body></html>',
    ]
    path = OUT / '研究报告.html'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'入口页 -> {path}')


def main():
    if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(parents=True, exist_ok=True)
    for name in STEPS:
        print(f'\n===== {name} =====')
        runpy.run_path(str(ROOT / name), run_name='__main__')
    write_index()
    print('三份报告已写入课题研究结论产出')


if __name__ == '__main__':
    main()
