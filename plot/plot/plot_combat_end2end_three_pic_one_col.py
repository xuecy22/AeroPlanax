import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from tensorboard.backend.event_processing import event_accumulator

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

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
def process_single_run_data(data, window_size=5, scale=1.0):
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
                event_accumulator.SCALARS: 0,
            }
        )
        ea.Reload()
        
        if tag not in ea.scalars.Keys():
            print(f"Could not find tag {tag} in {log_dir}")
            return None
        
        # 获取数据并转换为DataFrame
        events = ea.Scalars(tag)
        data = pd.DataFrame([(e.step, e.value) for e in events], 
                           columns=['step', 'value'])
        return data
    except Exception as e:
        print(f"Error processing {log_dir}: {e}")
        return None

def plot_end_to_end_selfplay():
    # 配置不同规模的实验路径 - End to End
    end_to_end_experiments = {
        "1v1": {
            "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent2_selfplay_seed0/logs"],
            "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent2_selfplay_seed10/logs"],
            "Seed 20": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent2_selfplay_seed20/logs"],
            "Seed 30": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent2_selfplay_seed30/logs"],
            "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent2_selfplay_seed42/logs"],
        },
        "2v2": {
            "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent4_selfplay_seed0/logs"],
            "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent4_selfplay_seed10/logs"],
            "Seed 20": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent4_selfplay_seed20/logs"],
            "Seed 30": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent4_selfplay_seed30/logs"],
            "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent4_selfplay_seed42/logs"],
        },
        "5v5": {
            "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent10_selfplay_seed0/logs"],
            "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent10_selfplay_seed10/logs"],
            "Seed 20": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent10_selfplay_seed20/logs"],
            "Seed 30": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent10_selfplay_seed30/logs"],
            "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent10_selfplay_seed42/logs"],
        },
    }
    
    # 要绘制的指标
    metric = "eval/episodic_return"
    
    # 设置绘图风格
    sns.set_theme(
        style="darkgrid",
        font_scale=1.2,
    )
    
    # 创建一行三列的子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300, sharey=False)
    
    # 设置每个规模的颜色
    scale_colors = {
        "1v1": "red",
        "2v2": "green",
        "5v5": "dodgerblue"
    }
    
    # 存储所有子图的y轴范围，用于后续统一
    y_ranges = []
    
    # 第一轮循环：处理数据并计算每个图的范围
    for i, scale in enumerate(["1v1", "2v2", "5v5"]):
        e2e_data = end_to_end_experiments[scale]
        ax = axes[i]
        
        # 处理End-to-End数据
        all_curves = []
        min_length = float('inf')
        
        for seed_name, log_dirs in e2e_data.items():
            seed_curves = []
            
            for log_dir in log_dirs:
                data = read_tensorboard_data(log_dir, metric)
                if data is None or data.empty:
                    continue
                
                _, smoothed_data = process_single_run_data(data['value'].values, window_size=5)
                seed_curves.append(smoothed_data)
                min_length = min(min_length, len(smoothed_data))
            
            if seed_curves:
                # 如果有多条曲线，取平均
                if len(seed_curves) > 1:
                    # 截断到相同长度
                    seed_curves = [curve[:min_length] for curve in seed_curves]
                    seed_curve = np.mean(np.array(seed_curves), axis=0)
                else:
                    seed_curve = seed_curves[0][:min_length]
                
                all_curves.append(seed_curve)
        
        # 如果没有有效数据，跳过此规模
        if not all_curves:
            continue
            
        # 截断到相同长度
        all_curves = [curve[:min_length] for curve in all_curves]
        
        # 转换为numpy数组并计算统计量
        all_curves = np.array(all_curves)
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        
        # 确定y轴范围
        y_min = np.min(mean_curve - std_curve)
        y_max = np.max(mean_curve + std_curve)
        
        # 添加一些边距
        y_range = y_max - y_min
        y_min = y_min - 0.1 * y_range
        y_max = y_max + 0.1 * y_range
        
        y_ranges.append((y_min, y_max))
    
    # 计算所有子图中的最大y轴范围
    max_range = 0
    for y_min, y_max in y_ranges:
        range_size = y_max - y_min
        max_range = max(max_range, range_size)
    
    # 调整y轴范围，保持中心点不变，但范围大小统一
    adjusted_y_ranges = []
    for y_min, y_max in y_ranges:
        center = (y_max + y_min) / 2
        half_range = max_range / 2
        adjusted_y_ranges.append((center - half_range, center + half_range))
    
    # 第二轮循环：实际绘制图形
    for i, scale in enumerate(["1v1", "2v2", "5v5"]):
        e2e_data = end_to_end_experiments[scale]
        ax = axes[i]
        
        # 处理End-to-End数据
        all_curves = []
        min_length = float('inf')
        
        for seed_name, log_dirs in e2e_data.items():
            seed_curves = []
            
            for log_dir in log_dirs:
                data = read_tensorboard_data(log_dir, metric)
                if data is None or data.empty:
                    continue
                
                _, smoothed_data = process_single_run_data(data['value'].values, window_size=5)
                seed_curves.append(smoothed_data)
                min_length = min(min_length, len(smoothed_data))
            
            if seed_curves:
                # 截断到相同长度
                seed_curves = [curve[:min_length] for curve in seed_curves]
                
                # 如果有多条曲线，取平均
                if len(seed_curves) > 1:
                    seed_curve = np.mean(np.array(seed_curves), axis=0)
                else:
                    seed_curve = seed_curves[0]
                
                all_curves.append(seed_curve)
        
        # 如果没有有效数据，跳过此规模
        if not all_curves:
            continue
            
        # 截断到相同长度
        all_curves = [curve[:min_length] for curve in all_curves]
        
        # 转换为numpy数组并计算统计量
        all_curves = np.array(all_curves)
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        
        # 创建x轴 - 这里将横坐标乘以3
        x = np.arange(0, min_length) * (300*1000/1e6) * 3
        
        # 绘制曲线
        ax.plot(x, mean_curve, color=scale_colors[scale], linewidth=2)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, 
                        color=scale_colors[scale], alpha=0.2)
        
        # 设置统一的y轴范围
        ax.set_ylim(adjusted_y_ranges[i])
        
        # 设置标题和标签
        ax.set_title(f"{scale} End-to-End SelfPlay", fontsize=14)
        ax.set_xlabel("Million Environment Steps", fontsize=12)
        if i == 0:  # 只在第一个图上添加y轴标签
            ax.set_ylabel("Average Reward", fontsize=12)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # 设置整体标题
    fig.suptitle("End-to-End SelfPlay Combat Task Performance", fontsize=16, y=1.05)
    
    # 调整布局并保存
    plt.tight_layout()
    output_dir = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/separate_three_pic_one_col(end2end_selfplay)/"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(f"{output_dir}end_to_end_selfplay.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}end_to_end_selfplay.pdf", bbox_inches="tight")
    print(f"End-to-End SelfPlay plot saved as {output_dir}end_to_end_selfplay.png and .pdf")
    
    plt.show()

if __name__ == "__main__":
    plot_end_to_end_selfplay()

# import os
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import matplotlib as mpl
# from tensorboard.backend.event_processing import event_accumulator

# # 设置英文字体
# plt.rcParams['font.sans-serif'] = ['Arial']
# plt.rcParams['axes.unicode_minus'] = False

# # 从tensorboard日志中读取数据并显示数据点数量
# def read_tensorboard_data(log_dir, tag, verbose=True):
#     try:
#         ea = event_accumulator.EventAccumulator(
#             log_dir,
#             size_guidance={
#                 event_accumulator.SCALARS: 0,
#             }
#         )
#         ea.Reload()
        
#         if tag not in ea.scalars.Keys():
#             if verbose:
#                 print(f"Could not find tag {tag} in {log_dir}")
#             return None
        
#         # 获取数据并转换为DataFrame
#         events = ea.Scalars(tag)
#         data = pd.DataFrame([(e.step, e.value) for e in events], 
#                            columns=['step', 'value'])
        
#         if verbose:
#             print(f"Log: {os.path.basename(log_dir)} contains {len(data)} data points")
#             # 添加起始步数和结束步数信息
#             if len(data) > 0:
#                 start_step = data['step'].iloc[0]
#                 end_step = data['step'].iloc[-1]
#                 print(f"  - Step range: {start_step} to {end_step}")
                
#         return data
#     except Exception as e:
#         if verbose:
#             print(f"Error processing {log_dir}: {e}")
#         return None

# def count_tensorboard_data_points():
#     # 配置不同规模的实验路径 - End to End
#     end_to_end_experiments = {
#         "1v1": {
#             "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent2_selfplay_seed0/logs"],
#             "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent2_selfplay_seed10/logs"],
#             "Seed 20": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent2_selfplay_seed20/logs"],
#             "Seed 30": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent2_selfplay_seed30/logs"],
#             "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent2_selfplay_seed42/logs"],
#         },
#         "2v2": {
#             "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent4_selfplay_seed0/logs"],
#             "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent4_selfplay_seed10/logs"],
#             "Seed 20": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent4_selfplay_seed20/logs"],
#             "Seed 30": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent4_selfplay_seed30/logs"],
#             "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent4_selfplay_seed42/logs"],
#         },
#         "5v5": {
#             "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent10_selfplay_seed0/logs"],
#             "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent10_selfplay_seed10/logs"],
#             "Seed 20": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent10_selfplay_seed20/logs"],
#             "Seed 30": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent10_selfplay_seed30/logs"],
#             "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent10_selfplay_seed42/logs"],
#         },
#     }
    
#     # 要绘制的指标
#     metric = "eval/episodic_return"
    
#     print("\n=== Data Point Count for End-to-End SelfPlay ===\n")
    
#     # 循环处理每个规模
#     for scale, seeds_data in end_to_end_experiments.items():
#         print(f"\n=== {scale} Scale ===")
        
#         # 处理每个种子
#         for seed_name, log_dirs in seeds_data.items():
#             print(f"\n{seed_name}:")
            
#             # 处理每个日志目录
#             for log_dir in log_dirs:
#                 read_tensorboard_data(log_dir, metric)

# if __name__ == "__main__":
#     count_tensorboard_data_points()