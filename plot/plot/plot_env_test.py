#!/usr/bin/env python3
"""
Log–log comparison of different flight simulators

‑ env_num: number of parallel instances
‑ time to finish 10 s (500 steps): seconds
‑ GPU memory: MB
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ------------------------------------------------------------------
# 1. Raw data -------------------------------------------------------
# ------------------------------------------------------------------
env_num = np.array([1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000])

# ---- Planax (your measured numbers)
planax_time = np.array([4.365, 3.965, 4.03, 4.275, 4.73, 6.7, 33.6, 245.5])
planax_mem  = np.array([3.0, 3.0, 3.2, 3.5, 4, 50, 622, 6228])

# ---- Reference simulators (approximated from the sample plot)
jsbsim_time   = np.array([0.02, 0.15, 1.5, 15, 150, 1_500, 15_000, 150_000])
jsbsim_mem    = np.full_like(env_num, 7.0, dtype=float)

ardupilot_time = np.array([0.008, 0.07, 0.7, 7, 70, 700, 7_000, 70_000])
ardupilot_mem  = np.full_like(env_num, 6.5, dtype=float)

xplane_time   = np.array([0.1, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000])
xplane_mem    = np.full_like(env_num, 7.3, dtype=float)

# ------------------------------------------------------------------
# 2. Plotting -------------------------------------------------------
# ------------------------------------------------------------------
plt.figure(figsize=(13, 6))

# ---- (a) Simulation‑time subplot
ax1 = plt.subplot(1, 2, 1)
ax1.loglog(env_num, planax_time,    marker='s', color='royalblue', label='Planax')
ax1.loglog(env_num, jsbsim_time,    marker='^', color='crimson',   label='JSBSim')
ax1.loglog(env_num, ardupilot_time, marker='o', color='forestgreen', label='Ardupilot')
ax1.loglog(env_num, xplane_time,    marker='D', color='darkorange', label='X‑Plane')

ax1.set_title('Log–Log Plot of Instances vs. Simulation Time')
ax1.set_xlabel('Number of Instances')
ax1.set_ylabel('Time of Performing 10 s Simulation (s)')
ax1.grid(True, which='both', ls='--', lw=0.5)
ax1.legend()
ax1.xaxis.set_major_formatter(ScalarFormatter())
# 修改横坐标为10的幂次
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))


# ---- (b) GPU‑memory subplot
ax2 = plt.subplot(1, 2, 2)
ax2.loglog(env_num, planax_mem,     marker='s', color='royalblue', label='Planax')
ax2.loglog(env_num, jsbsim_mem,     marker='^', color='crimson',   label='JSBSim')
ax2.loglog(env_num, ardupilot_mem,  marker='o', color='forestgreen', label='Ardupilot')
ax2.loglog(env_num, xplane_mem,     marker='D', color='darkorange', label='X‑Plane')

ax2.set_title('Log–Log Plot of Instances vs. GPU Memory')
ax2.set_xlabel('Number of Instances')
ax2.set_ylabel('GPU Memory of Performing 10 s Simulation (MB)')
ax2.grid(True, which='both', ls='--', lw=0.5)
ax2.legend()
ax2.xaxis.set_major_formatter(ScalarFormatter())
# 修改横坐标为10的幂次
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))


# Adjust layout and save
plt.tight_layout()
output_dir = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/env_test/"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}env test.png", bbox_inches="tight")
plt.savefig(f"{output_dir}env test.pdf", bbox_inches="tight")
print(f"\nChart saved as {output_dir}env test.png and {output_dir}env test.pdf")
plt.show()
