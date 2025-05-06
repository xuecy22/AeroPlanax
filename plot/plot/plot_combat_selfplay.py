# import os
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from tensorboard.backend.event_processing import event_accumulator

# # 定义平滑函数
# def smoother(x, a=0.9, w=1, mode="window"):
#     if mode == "window":
#         y = []
#         for i in range(len(x)):
#             y.append(np.mean(x[max(i - w, 0):i + 1]))
#     elif mode == "moving":
#         y = [x[0]]
#         for i in range(1, len(x)):
#             y.append((1 - a) * x[i] + a * y[i - 1])
#     else:
#         raise NotImplementedError
#     return y

# # 处理单个数据集并应用平滑
# def process_single_run_data(data, window_size=80, scale=1.0):
#     # 应用平滑
#     smoothed_data = smoother(np.asarray(data), w=window_size, mode="window")
#     x = np.arange(0, len(smoothed_data)) * scale
#     return x, smoothed_data

# # 从tensorboard日志中读取数据
# def read_tensorboard_data(log_dir, tag):
#     try:
#         ea = event_accumulator.EventAccumulator(
#             log_dir,
#             size_guidance={
#                 event_accumulator.SCALARS: 0,  # 加载所有scalars
#             }
#         )
#         ea.Reload()
        
#         if tag not in ea.scalars.Keys():
#             print(f"找不到标签 {tag} 在 {log_dir}")
#             return None
        
#         # 获取数据并转换为DataFrame
#         events = ea.Scalars(tag)
#         data = pd.DataFrame([(e.step, e.value) for e in events], 
#                            columns=['step', 'value'])
#         return data
#     except Exception as e:
#         print(f"处理 {log_dir} 时出错: {e}")
#         return None

# def plot_combat_performance():
#     # 配置不同规模的实验路径，按规模分组（用于后面合并同一规模的不同seed）
#     experiments_by_scale = {
#         "1v1": {
#             "种子 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed0/logs"],
#             "种子 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed10/logs"],
#             "种子 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed42/logs"],
#         },
#         "2v2": {
#             "种子 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed0/logs"],
#             "种子 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed10/logs"],
#             "种子 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed42/logs"],
#         },
#         "5v5": {
#             "种子 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed0/logs"],
#             "种子 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed10/logs"],
#             "种子 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed42/logs"],
#         },
#     }
    
#     # 要绘制的指标
#     metric = "eval/episodic_return"
    
#     # 设置绘图风格
#     sns.set_theme(
#         style="darkgrid",
#         font_scale=1.2,
#         rc={"figure.figsize": (12, 8)}
#     )
    
#     # 创建单张图表
#     fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
#     # 为不同规模选择不同颜色
#     scales = ["1v1", "2v2", "5v5"]
#     colors = sns.color_palette("viridis", len(scales))
    
#     # 创建颜色映射
#     color_map = {scale: color for scale, color in zip(scales, colors)}
    
#     # 存储最终性能数据以便后续总结
#     final_performance = {}
    
#     # 处理每个规模的实验
#     for scale, seeds_data in experiments_by_scale.items():
#         print(f"\n处理 {scale} 规模的数据")
        
#         # 存储这个规模下所有种子的曲线
#         all_scale_curves = []
#         min_length = float('inf')
        
#         # 处理每个种子的数据
#         for seed_name, log_dirs in seeds_data.items():
#             # 处理每个日志目录
#             for log_dir in log_dirs:
#                 data = read_tensorboard_data(log_dir, metric)
#                 if data is None or data.empty:
#                     print(f"警告: 无法读取 {log_dir} 的数据")
#                     continue
                
#                 # 平滑处理
#                 _, smoothed_data = process_single_run_data(data['value'].values, window_size=80)
#                 all_scale_curves.append(smoothed_data)
                
#                 # 记录最短曲线长度
#                 min_length = min(min_length, len(smoothed_data))
        
#         # 如果没有有效数据，跳过此规模
#         if not all_scale_curves:
#             print(f"跳过 {scale}: 没有有效数据")
#             continue
            
#         # 截断所有曲线到相同长度
#         all_scale_curves = [curve[:min_length] for curve in all_scale_curves]
        
#         # 转换为numpy数组并计算统计量
#         all_scale_curves = np.array(all_scale_curves)
#         mean_curve = np.mean(all_scale_curves, axis=0)
#         std_curve = np.std(all_scale_curves, axis=0)
        
#         # 创建x轴(环境步数，单位为百万)
#         x = np.arange(0, min_length) * (300*1000/1e6)  # 假设每300k步记录一次
        
#         # 绘制均值曲线
#         ax.plot(x, mean_curve, color=color_map[scale], label=f"{scale}", linewidth=2)
        
#         # 添加标准差区域
#         ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, 
#                        color=color_map[scale], alpha=0.2)
        
#         # 记录最终性能
#         final_perf = mean_curve[-1]
#         final_std = std_curve[-1]
#         final_performance[scale] = (final_perf, final_std)
#         print(f"{scale} 最终性能: {final_perf:.2f} ± {final_std:.2f}")
    
#     # 设置图表标题和标签
#     ax.set_title("Self-Play Training Performance Comparison", fontsize=16)
#     ax.set_xlabel("Million Environment Steps", fontsize=14)
#     ax.set_ylabel("Average Episode Return", fontsize=14)
#     ax.legend(loc="best", fontsize=12)
    
#     # 添加网格
#     ax.grid(True, linestyle='--', alpha=0.7)
    
#     # 调整布局并保存
#     plt.tight_layout()
#     output_dir = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat/selfplay/"
#     os.makedirs(output_dir, exist_ok=True)
    
#     plt.savefig(f"{output_dir}combat_selfplay_comparison.png", bbox_inches="tight")
#     plt.savefig(f"{output_dir}combat_selfplay_comparison.pdf", bbox_inches="tight")
#     print(f"\n图表已保存为 {output_dir}combat_selfplay_comparison.png 和 .pdf")
    
#     # 输出最终性能总结
#     print("\n最终性能总结:")
#     for scale, (perf, std) in final_performance.items():
#         print(f"{scale}: {perf:.2f} ± {std:.2f}")
    
#     plt.show()

# if __name__ == "__main__":
#     plot_combat_performance()


###############################################################################################################################

# import os
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from tensorboard.backend.event_processing import event_accumulator

# # 定义平滑函数
# def smoother(x, a=0.9, w=1, mode="window"):
#     if mode == "window":
#         y = []
#         for i in range(len(x)):
#             y.append(np.mean(x[max(i - w, 0):i + 1]))
#     elif mode == "moving":
#         y = [x[0]]
#         for i in range(1, len(x)):
#             y.append((1 - a) * x[i] + a * y[i - 1])
#     else:
#         raise NotImplementedError
#     return y

# # 处理单个数据集并应用平滑
# def process_single_run_data(data, window_size=80, scale=1.0):
#     # 应用平滑
#     smoothed_data = smoother(np.asarray(data), w=window_size, mode="window")
#     x = np.arange(0, len(smoothed_data)) * scale
#     return x, smoothed_data

# # 从tensorboard日志中读取数据
# def read_tensorboard_data(log_dir, tag):
#     try:
#         ea = event_accumulator.EventAccumulator(
#             log_dir,
#             size_guidance={
#                 event_accumulator.SCALARS: 0,  # 加载所有scalars
#             }
#         )
#         ea.Reload()
        
#         if tag not in ea.scalars.Keys():
#             print(f"找不到标签 {tag} 在 {log_dir}")
#             return None
        
#         # 获取数据并转换为DataFrame
#         events = ea.Scalars(tag)
#         data = pd.DataFrame([(e.step, e.value) for e in events], 
#                            columns=['step', 'value'])
#         return data
#     except Exception as e:
#         print(f"处理 {log_dir} 时出错: {e}")
#         return None

# def plot_combat_performance():
#     # 配置不同规模的实验路径，按规模分组
#     experiments_by_scale = {
#         "1v1": {
#             "seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed0/logs"],
#             "seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed10/logs"],
#             "seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed42/logs"],
#         },
#         "2v2": {
#             "seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed0/logs"],
#             "seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed10/logs"],
#             "seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed42/logs"],
#         },
#         "5v5": {
#             "seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed0/logs"],
#             "seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed10/logs"],
#             "seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed42/logs"],
#         },
#     }
    
#     # 要绘制的指标
#     metric = "eval/episodic_return"
    
#     # 设置绘图风格
#     sns.set_theme(
#         style="darkgrid",
#         font_scale=1.2,
#         rc={"figure.figsize": (16, 12)}
#     )
    
#     # 创建三个子图，每个规模一个
#     fig, axes = plt.subplots(3, 1, figsize=(12, 15), dpi=300)
    
#     # 为不同种子选择不同颜色
#     seeds = ["seed 0", "seed 10", "seed 42"]
#     colors = sns.color_palette("viridis", len(seeds))
    
#     # 创建颜色映射
#     color_map = {seed: color for seed, color in zip(seeds, colors)}
    
#     # 存储最终性能数据以便后续总结
#     final_performance = {}
    
#     # 为每个规模创建子图
#     for i, (scale, seeds_data) in enumerate(experiments_by_scale.items()):
#         print(f"\n处理 {scale} 规模的数据")
#         ax = axes[i]
        
#         # 处理每个种子的数据
#         for seed_name, log_dirs in seeds_data.items():
#             all_seed_curves = []
#             min_length = float('inf')
            
#             # 处理每个日志目录
#             for log_dir in log_dirs:
#                 data = read_tensorboard_data(log_dir, metric)
#                 if data is None or data.empty:
#                     print(f"警告: 无法读取 {log_dir} 的数据")
#                     continue
                
#                 # 平滑处理
#                 _, smoothed_data = process_single_run_data(data['value'].values, window_size=80)
#                 all_seed_curves.append(smoothed_data)
                
#                 # 记录最短曲线长度
#                 min_length = min(min_length, len(smoothed_data))
        
#             # 如果没有有效数据，跳过此种子
#             if not all_seed_curves:
#                 print(f"跳过 {scale} {seed_name}: 没有有效数据")
#                 continue
                
#             # 截断所有曲线到相同长度
#             all_seed_curves = [curve[:min_length] for curve in all_seed_curves]
            
#             # 转换为numpy数组
#             all_seed_curves = np.array(all_seed_curves)
            
#             # 如果只有一条曲线，直接用它
#             if len(all_seed_curves) == 1:
#                 seed_curve = all_seed_curves[0]
#             else:
#                 # 否则取平均值
#                 seed_curve = np.mean(all_seed_curves, axis=0)
            
#             # 创建x轴(环境步数，单位为百万)
#             x = np.arange(0, min_length) * (300*1000/1e6)  # 假设每300k步记录一次
            
#             # 绘制种子曲线
#             ax.plot(x, seed_curve, label=seed_name, linewidth=2)
            
#             # 记录最终性能
#             final_perf = seed_curve[-1]
#             final_performance[f"{scale}-{seed_name}"] = final_perf
#             print(f"{scale} {seed_name} 最终性能: {final_perf:.2f}")
        
#         # 设置每个子图的标题和标签
#         ax.set_title("Self-Play Training Performance Comparison", fontsize=16)
#         ax.set_xlabel("Million Environment Steps", fontsize=14)
#         ax.set_ylabel("Average Episode Return", fontsize=14)
#         ax.legend(loc="best", fontsize=12)
#         ax.grid(True, linestyle='--', alpha=0.7)
    
#     # 调整布局并保存
#     plt.tight_layout()
#     output_dir = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat/selfplay"
#     os.makedirs(output_dir, exist_ok=True)
    
#     plt.savefig(f"{output_dir}combat_selfplay_per_scale.png", bbox_inches="tight")
#     plt.savefig(f"{output_dir}combat_selfplay_per_scale.pdf", bbox_inches="tight")
#     print(f"\n图表已保存为 {output_dir}combat_selfplay_per_scale.png 和 .pdf")
    
#     # 输出最终性能总结
#     print("\n最终性能总结:")
#     for config, perf in final_performance.items():
#         print(f"{config}: {perf:.2f}")
    
#     plt.show()

# if __name__ == "__main__":
#     plot_combat_performance()

###############################################################################################################################

# import os
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import matplotlib as mpl
# from tensorboard.backend.event_processing import event_accumulator

# # 设置支持中文显示
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# # 定义平滑函数
# def smoother(x, a=0.9, w=1, mode="window"):
#     if mode == "window":
#         y = []
#         for i in range(len(x)):
#             y.append(np.mean(x[max(i - w, 0):i + 1]))
#     elif mode == "moving":
#         y = [x[0]]
#         for i in range(1, len(x)):
#             y.append((1 - a) * x[i] + a * y[i - 1])
#     else:
#         raise NotImplementedError
#     return y

# # 处理单个数据集并应用平滑
# def process_single_run_data(data, window_size=80, scale=1.0):
#     # 应用平滑
#     smoothed_data = smoother(np.asarray(data), w=window_size, mode="window")
#     x = np.arange(0, len(smoothed_data)) * scale
#     return x, smoothed_data

# # 从tensorboard日志中读取数据
# def read_tensorboard_data(log_dir, tag):
#     try:
#         ea = event_accumulator.EventAccumulator(
#             log_dir,
#             size_guidance={
#                 event_accumulator.SCALARS: 0,  # 加载所有scalars
#             }
#         )
#         ea.Reload()
        
#         if tag not in ea.scalars.Keys():
#             print(f"找不到标签 {tag} 在 {log_dir}")
#             return None
        
#         # 获取数据并转换为DataFrame
#         events = ea.Scalars(tag)
#         data = pd.DataFrame([(e.step, e.value) for e in events], 
#                            columns=['step', 'value'])
#         return data
#     except Exception as e:
#         print(f"处理 {log_dir} 时出错: {e}")
#         return None

# def plot_combat_performance():
#     # 配置不同规模的实验路径，按规模分组
#     experiments_by_scale = {
#         "1v1": {
#             "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed0/logs"],
#             "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed10/logs"],
#             "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed42/logs"],
#         },
#         "2v2": {
#             "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed0/logs"],
#             "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed10/logs"],
#             "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed42/logs"],
#         },
#         "5v5": {
#             "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed0/logs"],
#             "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed10/logs"],
#             "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed42/logs"],
#         },
#     }
    
#     # 要绘制的指标
#     metric = "eval/episodic_return"
    
#     # 设置绘图风格
#     sns.set_theme(
#         style="darkgrid",
#         font_scale=1.2,
#         rc={"figure.figsize": (16, 12)}
#     )
    
#     # 创建三个子图，每个规模一个
#     fig, axes = plt.subplots(3, 1, figsize=(12, 15), dpi=300)
    
#     # 为不同规模选择不同颜色
#     scales = ["1v1", "2v2", "5v5"]
#     colors = sns.color_palette("viridis", len(scales))
    
#     # 创建颜色映射
#     color_map = {scale: color for scale, color in zip(scales, colors)}
    
#     # 存储最终性能数据以便后续总结
#     final_performance = {}
    
#     # 为每个规模创建子图
#     for i, (scale, seeds_data) in enumerate(experiments_by_scale.items()):
#         print(f"\n处理 {scale} 规模的数据")
#         ax = axes[i]
        
#         # 收集该规模下所有种子的数据
#         all_scale_curves = []
#         min_length = float('inf')
        
#         # 处理每个种子的数据
#         for seed_name, log_dirs in seeds_data.items():
#             # 处理每个日志目录
#             for log_dir in log_dirs:
#                 data = read_tensorboard_data(log_dir, metric)
#                 if data is None or data.empty:
#                     print(f"警告: 无法读取 {log_dir} 的数据")
#                     continue
                
#                 # 平滑处理
#                 _, smoothed_data = process_single_run_data(data['value'].values, window_size=80)
#                 all_scale_curves.append(smoothed_data)
                
#                 # 记录最短曲线长度
#                 min_length = min(min_length, len(smoothed_data))
        
#         # 如果没有有效数据，跳过此规模
#         if not all_scale_curves:
#             print(f"跳过 {scale}: 没有有效数据")
#             continue
            
#         # 截断所有曲线到相同长度
#         all_scale_curves = [curve[:min_length] for curve in all_scale_curves]
        
#         # 转换为numpy数组并计算统计量
#         all_scale_curves = np.array(all_scale_curves)
#         mean_curve = np.mean(all_scale_curves, axis=0)
#         std_curve = np.std(all_scale_curves, axis=0)
        
#         # 创建x轴(环境步数，单位为百万)
#         x = np.arange(0, min_length) * (300*1000/1e6)  # 假设每300k步记录一次
        
#         # 绘制均值曲线
#         ax.plot(x, mean_curve, color=color_map[scale], label=f"{scale}", linewidth=2)
        
#         # 添加标准差区域
#         ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, 
#                        color=color_map[scale], alpha=0.2)
        
#         # 记录最终性能
#         final_perf = mean_curve[-1]
#         final_std = std_curve[-1]
#         final_performance[scale] = (final_perf, final_std)
#         print(f"{scale} 最终性能: {final_perf:.2f} ± {final_std:.2f}")
        
#         # 设置每个子图的标题和标签
#         ax.set_title("Self-Play Training Performance Comparison", fontsize=16)
#         ax.set_xlabel("Million Environment Steps", fontsize=14)
#         ax.set_ylabel("Average Episode Return", fontsize=14)
#         ax.grid(True, linestyle='--', alpha=0.7)
    
#     # 调整布局并保存
#     plt.tight_layout()
#     output_dir = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat/selfplay/"
#     os.makedirs(output_dir, exist_ok=True)
    
#     plt.savefig(f"{output_dir}combat_selfplay_per_scale2.png", bbox_inches="tight")
#     plt.savefig(f"{output_dir}combat_selfplay_per_scale2.pdf", bbox_inches="tight")
#     print(f"\n图表已保存为 {output_dir}combat_selfplay_per_scale2.png 和 .pdf")
    
#     # 输出最终性能总结
#     print("\n最终性能总结:")
#     for scale, (perf, std) in final_performance.items():
#         print(f"{scale}: {perf:.2f} ± {std:.2f}")
    
#     plt.show()

# if __name__ == "__main__":
#     plot_combat_performance()

###############################################################################################################################
# import os
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import matplotlib as mpl
# from tensorboard.backend.event_processing import event_accumulator

# # 设置支持中文显示
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# # 定义平滑函数
# def smoother(x, a=0.9, w=1, mode="window"):
#     if mode == "window":
#         y = []
#         for i in range(len(x)):
#             y.append(np.mean(x[max(i - w, 0):i + 1]))
#     elif mode == "moving":
#         y = [x[0]]
#         for i in range(1, len(x)):
#             y.append((1 - a) * x[i] + a * y[i - 1])
#     else:
#         raise NotImplementedError
#     return y

# # 处理单个数据集并应用平滑
# def process_single_run_data(data, window_size=80, scale=1.0):
#     # 应用平滑
#     smoothed_data = smoother(np.asarray(data), w=window_size, mode="window")
#     x = np.arange(0, len(smoothed_data)) * scale
#     return x, smoothed_data

# # 从tensorboard日志中读取数据
# def read_tensorboard_data(log_dir, tag):
#     try:
#         ea = event_accumulator.EventAccumulator(
#             log_dir,
#             size_guidance={
#                 event_accumulator.SCALARS: 0,  # 加载所有scalars
#             }
#         )
#         ea.Reload()
        
#         if tag not in ea.scalars.Keys():
#             print(f"找不到标签 {tag} 在 {log_dir}")
#             return None
        
#         # 获取数据并转换为DataFrame
#         events = ea.Scalars(tag)
#         data = pd.DataFrame([(e.step, e.value) for e in events], 
#                            columns=['step', 'value'])
#         return data
#     except Exception as e:
#         print(f"处理 {log_dir} 时出错: {e}")
#         return None

# def plot_combat_performance():
#     # 配置不同规模的实验路径，按规模分组
#     experiments_by_scale = {
#         "1v1": {
#             "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed0/logs"],
#             "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed10/logs"],
#             "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed42/logs"],
#         },
#         "2v2": {
#             "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed0/logs"],
#             "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed10/logs"],
#             "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed42/logs"],
#         },
#         "5v5": {
#             "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed0/logs"],
#             "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed10/logs"],
#             "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed42/logs"],
#         },
#     }
    
#     # 要绘制的指标
#     metric = "eval/episodic_return"
    
#     # 设置绘图风格
#     sns.set_theme(
#         style="darkgrid",
#         font_scale=1.2,
#         rc={"figure.figsize": (16, 12)}
#     )
    
#     # 创建三个子图，每个规模一个
#     fig, axes = plt.subplots(3, 1, figsize=(12, 15), dpi=300)
    
#     # 为不同规模选择不同颜色
#     scales = ["1v1", "2v2", "5v5"]
#     colors = sns.color_palette("viridis", len(scales))
    
#     # 创建颜色映射
#     color_map = {scale: color for scale, color in zip(scales, colors)}
    
#     # 存储最终性能数据以便后续总结
#     final_performance = {}
    
#     # 为了统一y轴范围，先收集所有子图的y轴范围
#     y_ranges = []
    
#     # 第一次循环：处理数据并确定每个子图的y轴范围
#     for i, (scale, seeds_data) in enumerate(experiments_by_scale.items()):
#         print(f"\n处理 {scale} 规模的数据")
        
#         # 收集该规模下所有种子的数据
#         all_scale_curves = []
#         min_length = float('inf')
        
#         # 处理每个种子的数据
#         for seed_name, log_dirs in seeds_data.items():
#             # 处理每个日志目录
#             for log_dir in log_dirs:
#                 data = read_tensorboard_data(log_dir, metric)
#                 if data is None or data.empty:
#                     print(f"警告: 无法读取 {log_dir} 的数据")
#                     continue
                
#                 # 平滑处理
#                 _, smoothed_data = process_single_run_data(data['value'].values, window_size=80)
#                 all_scale_curves.append(smoothed_data)
                
#                 # 记录最短曲线长度
#                 min_length = min(min_length, len(smoothed_data))
        
#         # 如果没有有效数据，跳过此规模
#         if not all_scale_curves:
#             print(f"跳过 {scale}: 没有有效数据")
#             y_ranges.append((0, 1))  # 添加一个默认范围
#             continue
            
#         # 截断所有曲线到相同长度
#         all_scale_curves = [curve[:min_length] for curve in all_scale_curves]
        
#         # 转换为numpy数组并计算统计量
#         all_scale_curves = np.array(all_scale_curves)
#         mean_curve = np.mean(all_scale_curves, axis=0)
#         std_curve = np.std(all_scale_curves, axis=0)
        
#         # 确定该子图的y轴范围
#         y_min = np.min(mean_curve - std_curve)
#         y_max = np.max(mean_curve + std_curve)
        
#         # 添加一些边距（10%）
#         y_range = y_max - y_min
#         y_min = y_min - 0.1 * y_range
#         y_max = y_max + 0.1 * y_range
        
#         y_ranges.append((y_min, y_max))
    
#     # 计算所有子图中的最大y轴范围
#     max_range = 0
#     for y_min, y_max in y_ranges:
#         range_size = y_max - y_min
#         max_range = max(max_range, range_size)
    
#     # 调整y轴范围，保持中心点不变，但范围大小统一
#     adjusted_y_ranges = []
#     for y_min, y_max in y_ranges:
#         center = (y_max + y_min) / 2
#         half_range = max_range / 2
#         adjusted_y_ranges.append((center - half_range, center + half_range))
    
#     # 第二次循环：实际绘制图形
#     for i, (scale, seeds_data) in enumerate(experiments_by_scale.items()):
#         ax = axes[i]
        
#         # 收集该规模下所有种子的数据
#         all_scale_curves = []
#         min_length = float('inf')
        
#         # 处理每个种子的数据
#         for seed_name, log_dirs in seeds_data.items():
#             # 处理每个日志目录
#             for log_dir in log_dirs:
#                 data = read_tensorboard_data(log_dir, metric)
#                 if data is None or data.empty:
#                     continue
                
#                 # 平滑处理
#                 _, smoothed_data = process_single_run_data(data['value'].values, window_size=80)
#                 all_scale_curves.append(smoothed_data)
                
#                 # 记录最短曲线长度
#                 min_length = min(min_length, len(smoothed_data))
        
#         # 如果没有有效数据，跳过此规模
#         if not all_scale_curves:
#             continue
            
#         # 截断所有曲线到相同长度
#         all_scale_curves = [curve[:min_length] for curve in all_scale_curves]
        
#         # 转换为numpy数组并计算统计量
#         all_scale_curves = np.array(all_scale_curves)
#         mean_curve = np.mean(all_scale_curves, axis=0)
#         std_curve = np.std(all_scale_curves, axis=0)
        
#         # 创建x轴(环境步数，单位为百万)
#         x = np.arange(0, min_length) * (300*1000/1e6)  # 假设每300k步记录一次
        
#         # 绘制均值曲线
#         ax.plot(x, mean_curve, color=color_map[scale], label=f"{scale}", linewidth=2)
        
#         # 添加标准差区域
#         ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, 
#                        color=color_map[scale], alpha=0.2)
        
#         # 设置调整后的y轴范围
#         ax.set_ylim(adjusted_y_ranges[i])
        
#         # 记录最终性能
#         final_perf = mean_curve[-1]
#         final_std = std_curve[-1]
#         final_performance[scale] = (final_perf, final_std)
#         print(f"{scale} 最终性能: {final_perf:.2f} ± {final_std:.2f}")
        
#         # 设置每个子图的标题和标签
#         ax.set_title(f"{scale} combat task (SelfPlay) Training Performance", fontsize=16)
#         ax.set_xlabel("Million Environment Steps", fontsize=14)
#         ax.set_ylabel("Average Episode Return", fontsize=14)
#         ax.grid(True, linestyle='--', alpha=0.7)
    
#     # 调整布局并保存
#     plt.tight_layout()
#     output_dir = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat/selfplay/"
#     os.makedirs(output_dir, exist_ok=True)
    
#     plt.savefig(f"{output_dir}combat_selfplay_per_scale_unified.png", bbox_inches="tight")
#     plt.savefig(f"{output_dir}combat_selfplay_per_scale_unified.pdf", bbox_inches="tight")
#     print(f"\n图表已保存为 {output_dir}combat_selfplay_per_scale_unified.png 和 .pdf")
    
#     # 输出最终性能总结
#     print("\n最终性能总结:")
#     for scale, (perf, std) in final_performance.items():
#         print(f"{scale}: {perf:.2f} ± {std:.2f}")
    
#     plt.show()

# if __name__ == "__main__":
#     plot_combat_performance()

###############################################################################################################################
# import os
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import matplotlib as mpl
# from tensorboard.backend.event_processing import event_accumulator

# # 设置支持中文显示
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# # 定义平滑函数
# def smoother(x, a=0.9, w=1, mode="window"):
#     if mode == "window":
#         y = []
#         for i in range(len(x)):
#             y.append(np.mean(x[max(i - w, 0):i + 1]))
#     elif mode == "moving":
#         y = [x[0]]
#         for i in range(1, len(x)):
#             y.append((1 - a) * x[i] + a * y[i - 1])
#     else:
#         raise NotImplementedError
#     return y

# # 处理单个数据集并应用平滑
# def process_single_run_data(data, window_size=80, scale=1.0):
#     # 应用平滑
#     smoothed_data = smoother(np.asarray(data), w=window_size, mode="window")
#     x = np.arange(0, len(smoothed_data)) * scale
#     return x, smoothed_data

# # 从tensorboard日志中读取数据
# def read_tensorboard_data(log_dir, tag):
#     try:
#         ea = event_accumulator.EventAccumulator(
#             log_dir,
#             size_guidance={
#                 event_accumulator.SCALARS: 0,  # 加载所有scalars
#             }
#         )
#         ea.Reload()
        
#         if tag not in ea.scalars.Keys():
#             print(f"找不到标签 {tag} 在 {log_dir}")
#             return None
        
#         # 获取数据并转换为DataFrame
#         events = ea.Scalars(tag)
#         data = pd.DataFrame([(e.step, e.value) for e in events], 
#                            columns=['step', 'value'])
#         return data
#     except Exception as e:
#         print(f"处理 {log_dir} 时出错: {e}")
#         return None

# def plot_combat_performance():
#     # 配置不同规模的实验路径，按规模分组
#     experiments_by_scale = {
#         "1v1": {
#             "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed0/logs"],
#             "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed10/logs"],
#             "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed42/logs"],
#         },
#         "2v2": {
#             "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed0/logs"],
#             "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed10/logs"],
#             "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed42/logs"],
#         },
#         "5v5": {
#             "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed0/logs"],
#             "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed10/logs"],
#             "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed42/logs"],
#         },
#     }
    
#     # 要绘制的指标
#     metric = "eval/episodic_return"
    
#     # 设置绘图风格
#     sns.set_theme(
#         style="darkgrid",
#         font_scale=1.2,
#         rc={"figure.figsize": (14, 10)}
#     )
    
#     # 创建一个图表，包含3个子图
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    
#     # 为不同规模选择不同颜色
#     scales = ["1v1", "2v2", "5v5"]
#     colors = sns.color_palette("viridis", len(scales))
    
#     # 创建颜色映射
#     color_map = {scale: color for scale, color in zip(scales, colors)}
    
#     # 存储最终性能数据以便后续总结
#     final_performance = {}
    
#     # 为了统一y轴范围，先收集所有子图的y轴范围
#     y_ranges = []
    
#     # 第一次循环：处理数据并确定每个子图的y轴范围
#     for i, (scale, seeds_data) in enumerate(experiments_by_scale.items()):
#         print(f"\n处理 {scale} 规模的数据")
        
#         # 收集该规模下所有种子的数据
#         all_scale_curves = []
#         min_length = float('inf')
        
#         # 处理每个种子的数据
#         for seed_name, log_dirs in seeds_data.items():
#             # 处理每个日志目录
#             for log_dir in log_dirs:
#                 data = read_tensorboard_data(log_dir, metric)
#                 if data is None or data.empty:
#                     print(f"警告: 无法读取 {log_dir} 的数据")
#                     continue
                
#                 # 平滑处理
#                 _, smoothed_data = process_single_run_data(data['value'].values, window_size=80)
#                 all_scale_curves.append(smoothed_data)
                
#                 # 记录最短曲线长度
#                 min_length = min(min_length, len(smoothed_data))
        
#         # 如果没有有效数据，跳过此规模
#         if not all_scale_curves:
#             print(f"跳过 {scale}: 没有有效数据")
#             y_ranges.append((0, 1))  # 添加一个默认范围
#             continue
            
#         # 截断所有曲线到相同长度
#         all_scale_curves = [curve[:min_length] for curve in all_scale_curves]
        
#         # 转换为numpy数组并计算统计量
#         all_scale_curves = np.array(all_scale_curves)
#         mean_curve = np.mean(all_scale_curves, axis=0)
#         std_curve = np.std(all_scale_curves, axis=0)
        
#         # 确定该子图的y轴范围
#         y_min = np.min(mean_curve - std_curve)
#         y_max = np.max(mean_curve + std_curve)
        
#         # 添加一些边距（10%）
#         y_range = y_max - y_min
#         y_min = y_min - 0.1 * y_range
#         y_max = y_max + 0.1 * y_range
        
#         y_ranges.append((y_min, y_max))
    
#     # 计算所有子图中的最大y轴范围
#     max_range = 0
#     for y_min, y_max in y_ranges:
#         range_size = y_max - y_min
#         max_range = max(max_range, range_size)
    
#     # 调整y轴范围，保持中心点不变，但范围大小统一
#     adjusted_y_ranges = []
#     for y_min, y_max in y_ranges:
#         center = (y_max + y_min) / 2
#         half_range = max_range / 2
#         adjusted_y_ranges.append((center - half_range, center + half_range))
    
#     # 第二次循环：实际绘制图形
#     for i, (scale, seeds_data) in enumerate(experiments_by_scale.items()):
#         ax = axes[i]
        
#         # 收集该规模下所有种子的数据
#         all_scale_curves = []
#         min_length = float('inf')
        
#         # 处理每个种子的数据
#         for seed_name, log_dirs in seeds_data.items():
#             # 处理每个日志目录
#             for log_dir in log_dirs:
#                 data = read_tensorboard_data(log_dir, metric)
#                 if data is None or data.empty:
#                     continue
                
#                 # 平滑处理
#                 _, smoothed_data = process_single_run_data(data['value'].values, window_size=80)
#                 all_scale_curves.append(smoothed_data)
                
#                 # 记录最短曲线长度
#                 min_length = min(min_length, len(smoothed_data))
        
#         # 如果没有有效数据，跳过此规模
#         if not all_scale_curves:
#             continue
            
#         # 截断所有曲线到相同长度
#         all_scale_curves = [curve[:min_length] for curve in all_scale_curves]
        
#         # 转换为numpy数组并计算统计量
#         all_scale_curves = np.array(all_scale_curves)
#         mean_curve = np.mean(all_scale_curves, axis=0)
#         std_curve = np.std(all_scale_curves, axis=0)
        
#         # 创建x轴(环境步数，单位为百万)
#         x = np.arange(0, min_length) * (300*1000/1e6)  # 假设每300k步记录一次
        
#         # 绘制均值曲线
#         ax.plot(x, mean_curve, color=color_map[scale], label=f"{scale}", linewidth=2)
        
#         # 添加标准差区域
#         ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, 
#                        color=color_map[scale], alpha=0.2)
        
#         # 设置调整后的y轴范围
#         ax.set_ylim(adjusted_y_ranges[i])
        
#         # 记录最终性能
#         final_perf = mean_curve[-1]
#         final_std = std_curve[-1]
#         final_performance[scale] = (final_perf, final_std)
#         print(f"{scale} 最终性能: {final_perf:.2f} ± {final_std:.2f}")
        
#         # 设置每个子图的标题和标签
#         ax.set_title(f"{scale} Combat Task (SelfPlay)", fontsize=16)
#         ax.set_xlabel("Million Environment Steps", fontsize=14)
#         ax.set_ylabel("Average Episode Return", fontsize=14)
#         ax.grid(True, linestyle='--', alpha=0.7)
        
#         # 添加图例
#         ax.legend(loc="lower right", fontsize=12)
    
#     # 整体标题
#     fig.suptitle("Combat Task (SelfPlay) Training Performance", fontsize=18, y=1.05)
    
#     # 调整布局并保存
#     plt.tight_layout()
#     output_dir = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat/selfplay/"
#     os.makedirs(output_dir, exist_ok=True)
    
#     plt.savefig(f"{output_dir}combat_selfplay_horizontal.png", bbox_inches="tight")
#     plt.savefig(f"{output_dir}combat_selfplay_horizontal.pdf", bbox_inches="tight")
#     print(f"\n图表已保存为 {output_dir}combat_selfplay_horizontal.png 和 .pdf")
    
#     # 输出最终性能总结
#     print("\n最终性能总结:")
#     for scale, (perf, std) in final_performance.items():
#         print(f"{scale}: {perf:.2f} ± {std:.2f}")
    
#     plt.show()

# if __name__ == "__main__":
#     plot_combat_performance()

###############################################################################################################################

import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from tensorboard.backend.event_processing import event_accumulator

# 设置支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 定义平滑函数
def smoother(x, a=0.9, w=1, mode="window"):
    if mode == "window":
        y = []
        for i in range(len(x)):
            y.append(np.mean(x[max(i - w, 0):i + 1]))
    elif mode == "moving":
        y = [x[0]]
        for i in range(1, len(x)):
            y.append((1 - a) * x[i] + a * y[i - 1])
    else:
        raise NotImplementedError
    return y

# 处理单个数据集并应用平滑
def process_single_run_data(data, window_size=80, scale=1.0):
    # 应用平滑
    smoothed_data = smoother(np.asarray(data), w=window_size, mode="window")
    x = np.arange(0, len(smoothed_data)) * scale
    return x, smoothed_data

# 从tensorboard日志中读取数据
def read_tensorboard_data(log_dir, tag):
    try:
        ea = event_accumulator.EventAccumulator(
            log_dir,
            size_guidance={
                event_accumulator.SCALARS: 0,  # 加载所有scalars
            }
        )
        ea.Reload()
        
        if tag not in ea.scalars.Keys():
            print(f"找不到标签 {tag} 在 {log_dir}")
            return None
        
        # 获取数据并转换为DataFrame
        events = ea.Scalars(tag)
        data = pd.DataFrame([(e.step, e.value) for e in events], 
                           columns=['step', 'value'])
        return data
    except Exception as e:
        print(f"处理 {log_dir} 时出错: {e}")
        return None

def plot_combat_performance():
    # 配置不同规模的实验路径，按规模分组
    experiments_by_scale = {
        "1v1": {
            "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed0/logs"],
            "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed10/logs"],
            "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent2_selfplay_seed42/logs"],
        },
        "2v2": {
            "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed0/logs"],
            "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed10/logs"],
            "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent4_selfplay_seed42/logs"],
        },
        "5v5": {
            "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed0/logs"],
            "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed10/logs"],
            "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/results/combat_selfplay/combat_agent10_selfplay_seed42/logs"],
        },
    }
    
    # 要绘制的指标
    metric = "eval/episodic_return"
    
    # 设置绘图风格
    sns.set_theme(
        style="darkgrid",
        font_scale=1.2,
        rc={"figure.figsize": (12, 8)}
    )
    
    # 创建单个图表
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # 为不同规模选择不同颜色
    scales = ["1v1", "2v2", "5v5"]
    colors = sns.color_palette("viridis", len(scales))
    
    # 创建颜色映射
    color_map = {scale: color for scale, color in zip(scales, colors)}
    
    # 存储最终性能数据以便后续总结
    final_performance = {}
    
    # 处理每个规模的实验
    for scale, seeds_data in experiments_by_scale.items():
        print(f"\n处理 {scale} 规模的数据")
        
        # 存储这个规模下所有种子的曲线
        all_scale_curves = []
        min_length = float('inf')
        
        # 处理每个种子的数据
        for seed_name, log_dirs in seeds_data.items():
            # 处理每个日志目录
            for log_dir in log_dirs:
                data = read_tensorboard_data(log_dir, metric)
                if data is None or data.empty:
                    print(f"警告: 无法读取 {log_dir} 的数据")
                    continue
                
                # 平滑处理
                _, smoothed_data = process_single_run_data(data['value'].values, window_size=80)
                all_scale_curves.append(smoothed_data)
                
                # 记录最短曲线长度
                min_length = min(min_length, len(smoothed_data))
        
        # 如果没有有效数据，跳过此规模
        if not all_scale_curves:
            print(f"跳过 {scale}: 没有有效数据")
            continue
            
        # 截断所有曲线到相同长度
        all_scale_curves = [curve[:min_length] for curve in all_scale_curves]
        
        # 转换为numpy数组并计算统计量
        all_scale_curves = np.array(all_scale_curves)
        mean_curve = np.mean(all_scale_curves, axis=0)
        std_curve = np.std(all_scale_curves, axis=0)
        
        # 创建x轴(环境步数，单位为百万)
        x = np.arange(0, min_length) * (300*1000/1e6)  # 假设每300k步记录一次
        
        # 绘制均值曲线
        ax.plot(x, mean_curve, color=color_map[scale], label=f"{scale}", linewidth=2)
        
        # 添加标准差区域
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, 
                       color=color_map[scale], alpha=0.2)
        
        # 记录最终性能
        final_perf = mean_curve[-1]
        final_std = std_curve[-1]
        final_performance[scale] = (final_perf, final_std)
        print(f"{scale} 最终性能: {final_perf:.2f} ± {final_std:.2f}")
    
    # 设置图表标题和标签
    ax.set_title("Combat Task (SelfPlay) Training Performance", fontsize=16)
    ax.set_xlabel("Million Environment Steps", fontsize=14)
    ax.set_ylabel("Average Reward", fontsize=14)
    # ax.legend(loc="best", fontsize=12)
    ax.legend(loc="lower right", fontsize=12)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局并保存
    plt.tight_layout()
    output_dir = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat/selfplay/"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(f"{output_dir}combat_selfplay_single_plot.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}combat_selfplay_single_plot.pdf", bbox_inches="tight")
    print(f"\n图表已保存为 {output_dir}combat_selfplay_single_plot.png 和 .pdf")
    
    # 输出最终性能总结
    print("\n最终性能总结:")
    for scale, (perf, std) in final_performance.items():
        print(f"{scale}: {perf:.2f} ± {std:.2f}")
    
    plt.show()

if __name__ == "__main__":
    plot_combat_performance()