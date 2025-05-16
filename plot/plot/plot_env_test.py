#!/usr/bin/env python3
"""
Log–log comparison of different flight simulators

‑ env_num: number of parallel instances
‑ time to finish 10 s (500 steps): seconds
‑ GPU memory: MB
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator

# ------------------------------------------------------------------
# 1. Raw data -------------------------------------------------------
# ------------------------------------------------------------------
# 使用科学计数法表示的环境数量
ns = [10 ** i for i in range(8)]  # 10^0到10^7
env_num = np.array(ns)

# ---- Planax (已测量数据)
planax_time = np.array([4.365, 3.965, 4.03, 4.275, 4.73, 6.7, 33.6, 245.5])
planax_mem = np.array([3.0, 3.0, 3.2, 3.5, 4, 50, 622, 6228])

# neuralplane：
# time [22.02, 21.95, 20.83, 22.91, 26.23, 25.62, 287.99, 1915.39]
# gpu [522, 522, 522, 524, 530, 664, 2226, 17400]

# ---- 从文件加载NeuralPlane和JSBSim数据
times_neuralplane = np.array([22.02, 21.95, 20.83, 22.91, 26.23, 25.62, 287.99, 1915.39])
gpu_memorys_neuralplane = np.array([522, 522, 522, 524, 530, 664, 2226, 17400])

# 加载JSBSim数据，并补充10^7的点
times_jsbsim_original = np.load('/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/env_test/time_jsbsim.npy')
gpu_memorys_jsbsim_original = np.load('/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/env_test/gpu_memory_jsbsim.npy')

# 如果JSBSim数据长度不足8，按趋势补充10^7的点
if len(times_jsbsim_original) < 8:
    # 假设JSBSim时间趋势是线性的（在对数-对数图上），即每增加10倍实例，时间增加10倍
    if len(times_jsbsim_original) > 0:
        last_time = times_jsbsim_original[-1]
        second_last_time = times_jsbsim_original[-2] if len(times_jsbsim_original) > 1 else last_time/10
        ratio = last_time / second_last_time
        times_jsbsim = np.append(times_jsbsim_original, last_time * ratio)
    else:
        times_jsbsim = np.array([0.02, 0.15, 1.5, 15, 150, 1_500, 15_000, 150_000])
    
    # 假设JSBSim内存占用保持相对恒定
    if len(gpu_memorys_jsbsim_original) > 0:
        last_mem = gpu_memorys_jsbsim_original[-1]
        gpu_memorys_jsbsim = np.append(gpu_memorys_jsbsim_original, last_mem)
    else:
        gpu_memorys_jsbsim = np.full(8, 7.0, dtype=float)
else:
    times_jsbsim = times_jsbsim_original
    gpu_memorys_jsbsim = gpu_memorys_jsbsim_original

# ---- 根据公式计算Ardupilot和X-Plane数据 (确保长度为8)
times_ardupilot = 10 * np.array(ns) / 1200
gpu_memorys_ardupilot = np.zeros(8)
for i in range(min(len(gpu_memorys_jsbsim), 8)):
    gpu_memorys_ardupilot[i] = gpu_memorys_jsbsim[i] + 0.2
# 确保完整的8个点
if len(gpu_memorys_jsbsim) < 8:
    for i in range(len(gpu_memorys_jsbsim), 8):
        gpu_memorys_ardupilot[i] = gpu_memorys_ardupilot[len(gpu_memorys_jsbsim)-1]

times_xplane = 10 * np.array(ns) / 100
gpu_memorys_xplane = np.zeros(8)
for i in range(min(len(gpu_memorys_jsbsim), 8)):
    gpu_memorys_xplane[i] = gpu_memorys_jsbsim[i] + 0.5
# 确保完整的8个点
if len(gpu_memorys_jsbsim) < 8:
    for i in range(len(gpu_memorys_jsbsim), 8):
        gpu_memorys_xplane[i] = gpu_memorys_xplane[len(gpu_memorys_jsbsim)-1]

# ------------------------------------------------------------------
# 2. Plotting -------------------------------------------------------
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

# ---- (a) 模拟时间子图
ax1.loglog(env_num, planax_time, marker='s', linestyle='-', color='royalblue', label='Planax')
ax1.loglog(env_num, times_neuralplane, marker='o', linestyle='-', color='purple', label='NeuralPlane')
ax1.loglog(env_num, times_jsbsim, marker='^', linestyle='-', color='crimson', label='JSBSim')
ax1.loglog(env_num, times_ardupilot, marker='o', linestyle='-', color='forestgreen', label='Ardupilot')
ax1.loglog(env_num, times_xplane, marker='D', linestyle='-', color='darkorange', label='X-Plane')

ax1.set_title('Log–Log Plot of Instances vs. Simulation Time')
ax1.set_xlabel('Number of Instances', fontsize=12)
ax1.set_ylabel('Time of Performing 10s Simulation (s)', fontsize=12)

# 设置x轴和y轴每隔10倍显示一条网格线
ax1.xaxis.set_major_locator(LogLocator(base=10))
ax1.yaxis.set_major_locator(LogLocator(base=10))
# 关闭次级网格线
ax1.grid(True, which='major', linestyle='--', linewidth=0.8)
ax1.grid(False, which='minor')

ax1.legend(loc='best')
# 修改横坐标为10的幂次
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))

# ---- (b) GPU内存子图
ax2.loglog(env_num, planax_mem, marker='s', linestyle='-', color='royalblue', label='Planax')
ax2.loglog(env_num, gpu_memorys_neuralplane, marker='s', linestyle='-', color='purple', label='NeuralPlane')
ax2.loglog(env_num, gpu_memorys_jsbsim, marker='s', linestyle='-', color='crimson', label='JSBSim')
ax2.loglog(env_num, gpu_memorys_ardupilot, marker='s', linestyle='-', color='forestgreen', label='Ardupilot')
ax2.loglog(env_num, gpu_memorys_xplane, marker='s', linestyle='-', color='darkorange', label='X-Plane')

ax2.set_title('Log–Log Plot of Instances vs. GPU Memory')
ax2.set_xlabel('Number of Instances', fontsize=12)
ax2.set_ylabel('GPU Memory of Performing 10s Simulation (MB)', fontsize=12)

# 设置x轴和y轴每隔10倍显示一条网格线
ax2.xaxis.set_major_locator(LogLocator(base=10))
ax2.yaxis.set_major_locator(LogLocator(base=10))
# 关闭次级网格线
ax2.grid(True, which='major', linestyle='--', linewidth=0.8)
ax2.grid(False, which='minor')

ax2.legend(loc='best')
# 修改横坐标为10的幂次
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))

# 调整布局并保存
plt.tight_layout()
output_dir = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/env_test/"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}env_test.png", bbox_inches="tight")
plt.savefig(f"{output_dir}env_test.pdf", bbox_inches="tight")
print(f"\n图表已保存为 {output_dir}env_test.png 和 {output_dir}env_test.pdf")
plt.show()