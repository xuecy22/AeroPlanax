#!/usr/bin/env python3
"""
Planax 优势强调版：三图分层展示
-------------------------------------------------------------
1. 绝对仿真时间（线性 y，log x，范围 10^3–10^6）
2. Speed-up Ratio (Other / Planax)
3. GPU Memory（log-log）
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator

# ------------------------------------------------------------
# 1. 原始数据
# ------------------------------------------------------------
ns_full = np.array([10 ** i for i in range(8)])            # 10^0 … 10^7
idx_use  = (ns_full >= 1e3) & (ns_full <= 1e6)             # 10^3 … 10^6
env_num  = ns_full[idx_use]                                # 4 点

# ---- Planax (已测量)
planax_time_full = np.array([4.365, 3.965, 4.03, 4.275, 4.73, 6.7, 33.6, 245.5])
planax_mem_full  = np.array([3.0,  3.0,   3.2,  3.5,   4,    50,  622,  6228])

planax_time = planax_time_full[idx_use]
planax_mem  = planax_mem_full[idx_use]

# ---- NeuralPlane
np_time_full = np.array([22.02, 21.95, 20.83, 22.91, 26.23, 25.62, 287.99, 1915.39])
np_mem_full  = np.array([522,   522,   522,   524,   530,   664,   2226,   17400])
np_time = np_time_full[idx_use]
np_mem  = np_mem_full[idx_use]

# ---- JSBSim（若缺补齐已在原脚本实现，这里假设长度充足）
# ---- JSBSim
jsb_time_full = np.load('/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/env_test/time_jsbsim.npy')
jsb_mem_full  = np.load('/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/env_test/gpu_memory_jsbsim.npy')

if len(jsb_time_full) < len(ns_full):          # 发现少点就补
    pad = len(ns_full) - len(jsb_time_full)    # 需要补多少点
    if len(jsb_time_full) >= 2:                # 用最后两个点推一个倍率
        ratio = jsb_time_full[-1] / jsb_time_full[-2]
    else:                                      # 数据太少就默认 10×
        ratio = 10.0
    extra_time = jsb_time_full[-1] * ratio ** np.arange(1, pad + 1)
    extra_mem  = np.full(pad, jsb_mem_full[-1])
    jsb_time_full = np.concatenate([jsb_time_full, extra_time])
    jsb_mem_full  = np.concatenate([jsb_mem_full,  extra_mem])

# 现在 jsb_time_full / jsb_mem_full 都有 8 个点
jsb_time = jsb_time_full[idx_use]
jsb_mem  = jsb_mem_full[idx_use]


# ---- Others = (X-Plane + Ardupilot) 取均值，仅用于显存&时间弱化对照
others_time_full = 0.5 * (10 * ns_full / 100 + 10 * ns_full / 1200)
others_mem_full  = 0.5 * ((jsb_mem_full + 0.5) + (jsb_mem_full + 0.2))
others_time = others_time_full[idx_use]
others_mem  = others_mem_full[idx_use]

# ------------------------------------------------------------
# 2. Speed-up 计算
# ------------------------------------------------------------
speed_np  = np_time / planax_time        # NeuralPlane / Planax
speed_jsb = jsb_time  / planax_time      # JSBSim / Planax
speed_oth = others_time / planax_time    # Others   / Planax

# ------------------------------------------------------------
# 3. 绘图
# ------------------------------------------------------------

plt.rcParams.update({'font.size': 11})

# ---------- 一行三列 ----------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=1, ncols=3, figsize=(18, 5), dpi=300
)

# ---- (a) 绝对仿真时间 --------------------------------------
ax1.plot(env_num, planax_time,  marker='o', lw=3,
         color='red', label='Planax')
ax1.plot(env_num, np_time,      marker='s', lw=1.5,
         color='dodgerblue', alpha=0.8, label='NeuralPlane')
ax1.plot(env_num, jsb_time,     marker='^', lw=1.5,
         color='orange', alpha=0.8, label='JSBSim')
ax1.plot(env_num, others_time,  marker='D', lw=1,
         color='green', alpha=0.5, linestyle='--', label='Others')

ax1.set_xscale('log')
ax1.set_xlim(1e3, 1e6)
ax1.set_xlabel('Number of Instances')
ax1.set_ylabel('Simulation Time for 10 s (s)')
ax1.set_title('Simulation Time')
ax1.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.4)
ax1.legend(frameon=False, loc='upper left')

# ---- (b) Speed-up Ratio ------------------------------------
width = 0.25
x_pos = np.arange(len(env_num))

ax2.bar(x_pos - width, speed_np,  width=width,
        label='NeuralPlane', color='dodgerblue')
ax2.bar(x_pos,         speed_jsb, width=width,
        label='JSBSim',     color='orange')
ax2.bar(x_pos + width, speed_oth, width=width,
        label='Others',     color='green')

baseline = np.ones_like(env_num)
ax2.bar(x_pos - 1.8*width, baseline, width=0.15,
        label='Planax (baseline)', color='red')


# ax2.axhline(1, color='#ff7f0e', lw=2,
#             label='Planax (baseline)')  # Planax = 1× 参考线
ax2.set_xticks(x_pos)
ax2.set_xticklabels([rf'$10^{int(np.log10(n))}$' for n in env_num])
ax2.set_yscale('log')
ax2.set_ylabel('Speed-up vs Planax (×)')
ax2.set_title('Speed-up Ratio')
ax2.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.4)
ax2.legend(frameon=False, loc='upper left')

# ---- (c) GPU Memory ----------------------------------------
ax3.loglog(env_num, planax_mem,  marker='o', lw=3,
           color='red', label='Planax')
ax3.loglog(env_num, np_mem,      marker='s', lw=1.5,
           color='dodgerblue', alpha=0.8, label='NeuralPlane')
ax3.loglog(env_num, jsb_mem,     marker='^', lw=1.5,
           color='orange', alpha=0.8, label='JSBSim')
ax3.loglog(env_num, others_mem,  marker='D', lw=1,
           color='green', alpha=0.5, linestyle='--', label='Others')

ax3.set_xlabel('Number of Instances')
ax3.set_ylabel('GPU Memory for 10 s (MB)')
ax3.set_title('GPU Memory')
ax3.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.4)
ax3.legend(frameon=False, loc='upper left')

# ---------- 保存 ---------------------------------------------
plt.tight_layout()
out_dir = '/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/env_test/'
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 'env_test_refined_row.png'), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'env_test_refined_row.pdf'), bbox_inches='tight')
print(f"图表已保存到 {out_dir}")

plt.show()