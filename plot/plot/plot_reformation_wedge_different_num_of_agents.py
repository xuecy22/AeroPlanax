import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

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

# 主函数
def plot_reformation_results():
    # 配置实验路径字典，以agent数量为索引
    experiments = {
        "Agent 2": [
            "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/wedge_different_num_of_agents/reformation_agent2_wedge_seed0/logs",
            "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/wedge_different_num_of_agents/reformation_agent2_wedge_seed10/logs",
            "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/wedge_different_num_of_agents/reformation_agent2_wedge_seed42/logs",
            "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/wedge_different_num_of_agents/reformation_agent2_wedge_seed20/logs",
            "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/wedge_different_num_of_agents/reformation_agent2_wedge_seed30/logs",
        ],
        "Agent 5": [
            "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/wedge_different_num_of_agents/reformation_agent5_wedge_seed0/logs",
            "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/wedge_different_num_of_agents/reformation_agent5_wedge_seed10/logs",
            "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/wedge_different_num_of_agents/reformation_agent5_wedge_seed42/logs",
            "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/wedge_different_num_of_agents/reformation_agent5_wedge_seed20/logs",
            "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/wedge_different_num_of_agents/reformation_agent5_wedge_seed30/logs",
        ],
        "Agent 10": [
            "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/wedge_different_num_of_agents/reformation_agent10_wedge_seed0/logs",
            "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/wedge_different_num_of_agents/reformation_agent10_wedge_seed10/logs",
            "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/wedge_different_num_of_agents/reformation_agent10_wedge_seed42/logs",
            "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/wedge_different_num_of_agents/reformation_agent10_wedge_seed20/logs",
            "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/wedge_different_num_of_agents/reformation_agent10_wedge_seed30/logs",
        ]
    }
    
    # 只绘制回合奖励指标
    metric = "eval/episodic_return"
    
    # 设置绘图风格
    sns.set_theme(
        style="darkgrid",
        font_scale=1.2,
        rc={"figure.figsize": (10, 6)}
    )
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = sns.color_palette("Set1", len(experiments))
    
    # 为每个agent配置处理数据
    for i, (agent_name, log_dirs) in enumerate(experiments.items()):
        print(f"\n处理 {agent_name} 的数据")
        
        # 存储所有种子的处理后数据
        all_seed_curves = []
        min_length = float('inf')
        
        # 读取和处理每个种子的数据
        for log_dir in log_dirs:
            data = read_tensorboard_data(log_dir, metric)
            if data is None or data.empty:
                print(f"警告: 无法读取 {log_dir} 的数据")
                continue
            
            # 平滑处理
            _, smoothed_data = process_single_run_data(data['value'].values, window_size=5)
            all_seed_curves.append(smoothed_data)
            
            # 记录最短数据长度
            min_length = min(min_length, len(smoothed_data))
        
        # 如果没有有效数据，跳过此agent
        if not all_seed_curves:
            print(f"跳过 {agent_name}: 没有有效数据")
            continue
            
        # 截断所有曲线到相同长度
        all_seed_curves = [curve[:min_length] for curve in all_seed_curves]
        
        # 转换为numpy数组并计算均值和标准差
        all_seed_curves = np.array(all_seed_curves)
        mean_curve = np.mean(all_seed_curves, axis=0)
        std_curve = np.std(all_seed_curves, axis=0)
        
        # 创建x轴（使用环境步数作为单位，单位为百万）
        x = np.arange(0, min_length) * (300*3000/1e6)
        
        # 绘制均值曲线
        ax.plot(x, mean_curve, color=colors[i], label=agent_name, linewidth=2)
        
        # 添加标准差阴影
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=colors[i], alpha=0.2)
        
        # 输出最终性能
        print(f"{agent_name} 最终值: {mean_curve[-1]:.2f} ± {std_curve[-1]:.2f}")
    
    # 设置图表标题和标签
    ax.set_title("Reformation Task (Wedge) Training Performance with Different Agent Numbers", fontsize=14)
    ax.set_xlabel("Million Environment Steps", fontsize=12)
    ax.set_ylabel("Average Reward", fontsize=12)
    ax.legend(loc="best", fontsize=12)
    
    # 显示网格
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局并保存
    plt.tight_layout()
    output_dir = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/wedge_different_num_of_agents/"
    plt.savefig(f"{output_dir}Reformation Task (Wedge) Training Performance with Different Agent Numbers.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}Reformation Task (Wedge) Training Performance with Different Agent Numbers.pdf", bbox_inches="tight")
    print(f"\n图表已保存为 {output_dir}Reformation Task (Wedge) Training Performance with Different Agent Numbers.png 和 {output_dir}Reformation Task (Wedge) Training Performance with Different Agent Numbers.pdf")
    plt.show()

# 执行主函数
if __name__ == "__main__":
    plot_reformation_results()