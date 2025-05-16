#!/usr/bin/env python3
"""
Three-panel comparison of flight simulators
  1. Simulation Time     (log-log)
  2. Speed-up Factor     (max_time / time_i)
  3. GPU Memory          (log-log)
"""

import os, numpy as np, matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

# ---------- 0 颜色 ----------
C_PLAN   = '#d62728'   # red
C_NEU    = '#2ca02c'   # green
C_JSB    = '#1f77b4'   # blue
C_OTH    = '#ff7f0e'   # orange
ALPHA    = 0.50

# ---------- 1 数据 ----------
ns = np.array([10 ** i for i in range(8)])      # 10^0 … 10^7

plan_time = np.array([4.365, 3.965, 4.03, 4.275, 4.73, 6.7, 33.6, 245.5])
plan_mem  = np.array([3.0, 3.0, 3.2, 3.5, 4, 50, 622, 6228])

neu_time  = np.array([22.02, 21.95, 20.83, 22.91, 26.23, 25.62, 287.99, 1915.39])
neu_mem   = np.array([522, 522, 522, 524, 530, 664, 2226, 17400])

jsb_time  = np.load('/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/env_test/time_jsbsim.npy')
jsb_mem   = np.load('/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/env_test/gpu_memory_jsbsim.npy')
if len(jsb_time) < len(ns):                     # 自动补齐
    pad, ratio = len(ns) - len(jsb_time), jsb_time[-1] / jsb_time[-2]
    jsb_time = np.concatenate([jsb_time, jsb_time[-1] * ratio ** np.arange(1, pad + 1)])
    jsb_mem  = np.concatenate([jsb_mem,  np.full(pad, jsb_mem[-1])])

oth_time  = 0.5 * (10 * ns / 100 + 10 * ns / 1200)
oth_mem   = 0.5 * ((jsb_mem + .5) + (jsb_mem + .2))

# ---------- 2′ Speed-up（NeuralPlane 为基准） ----------
idx = slice(3, 8)                       # 10^3 … 10^7 → 下标 3~7
base_time = neu_time[idx]               # NeuralPlane 时间

spd_plan = base_time / plan_time[idx]
spd_neu  = np.ones_like(base_time)      # 基准 = 1×
spd_jsb  = base_time / jsb_time[idx]
spd_oth  = base_time / oth_time[idx]

# ---------- 3 绘图 ----------
plt.rcParams.update({'font.size': 11})
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

# ---- 图 1 Simulation Time (log-log) -------------------------
ax1.loglog(ns, plan_time,  's-', lw=2.5, color=C_PLAN, label='Planax')
ax1.loglog(ns, neu_time,   'o-', lw=1.8, color=C_NEU,  label='NeuralPlane')
ax1.loglog(ns, jsb_time,   '^-', lw=1.8, color=C_JSB,  label='JSBSim')
ax1.loglog(ns, oth_time,   'D--',lw=1.5, color=C_OTH,  label='Others')
ax1.set_xlabel('Number of Instances'); ax1.set_ylabel('Simulation Time (s)')
ax1.set_title('Simulation Time (log-log)')
ax1.xaxis.set_major_locator(LogLocator(10)); ax1.yaxis.set_major_locator(LogLocator(10))
ax1.grid(True, which='major', ls='--', alpha=.4); ax1.legend(frameon=False, loc='upper left')

# ---------- 3′ Speed-up 子图重画 --------------------------
width = .18
x_pos = np.arange(idx.stop - idx.start)           # 0…4

ax2.bar(x_pos - 1.5*width, spd_plan, width,
        color=C_PLAN, ec=C_PLAN, alpha=ALPHA, label='Planax')
ax2.bar(x_pos - 0.5*width, spd_neu,  width,
        color=C_NEU,  ec=C_NEU,  alpha=ALPHA, label='NeuralPlane')
ax2.bar(x_pos + 0.5*width, spd_jsb,  width,
        color=C_JSB,  ec=C_JSB,  alpha=ALPHA, label='JSBSim')
ax2.bar(x_pos + 1.5*width, spd_oth,  width,
        color=C_OTH,  ec=C_OTH,  alpha=ALPHA, label='Others')

ax2.set_xticks(x_pos)
ax2.set_xticklabels([r'$10^{%d}$' % p for p in range(3, 8)])
ax2.set_xlabel('Number of Instances');
ax2.set_ylabel('Speed-up over NeuralPlane (×)')
ax2.set_title('Speed-up Ratio')
ax2.grid(True, axis='y', ls='--', alpha=.4)
ax2.legend(frameon=False, loc='upper left')

# ---- 图 3 GPU Memory (log-log) ------------------------------
ax3.loglog(ns, plan_mem, 's-', lw=2.5, color=C_PLAN, label='Planax')
ax3.loglog(ns, neu_mem,  'o-', lw=1.8, color=C_NEU,  label='NeuralPlane')
ax3.loglog(ns, jsb_mem,  '^-', lw=1.8, color=C_JSB,  label='JSBSim')
ax3.loglog(ns, oth_mem,  'D--',lw=1.5, color=C_OTH,  label='Others')
ax3.set_xlabel('Number of Instances'); ax3.set_ylabel('GPU Memory (MB)')
ax3.set_title('GPU Memory (log-log)')
ax3.xaxis.set_major_locator(LogLocator(10)); ax3.yaxis.set_major_locator(LogLocator(10))
ax3.grid(True, which='major', ls='--', alpha=.4); ax3.legend(frameon=False, loc='upper left')

plt.tight_layout()
out_dir = '/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/env_test/'
os.makedirs(out_dir, exist_ok=True)
plt.savefig(out_dir + 'env_test_refined_row.png', bbox_inches='tight')
plt.savefig(out_dir + 'env_test_refined_row.pdf', bbox_inches='tight')
print('图表已保存到', out_dir)
plt.show()
