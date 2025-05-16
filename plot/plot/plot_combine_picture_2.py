import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

def combine_images(image_paths, output_path, nrows=1, ncols=2, figsize=(16, 5)):
    """
    将多张图片并排放置，组成一张图
    
    参数:
        image_paths: 图片路径列表
        output_path: 输出图片路径
        nrows: 行数
        ncols: 列数
        figsize: 图片尺寸
    """
    # 创建图像
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows, ncols, figure=fig)
    
    # 加载并放置每张图片
    for i, img_path in enumerate(image_paths):
        if i >= nrows * ncols:
            print(f"Warning: Only {nrows*ncols} images can be displayed, ignoring remaining images")
            break
            
        row = i // ncols
        col = i % ncols
        
        # 读取图片
        img = mpimg.imread(img_path)
        
        # 在指定位置添加子图
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.axis('off')  # 关闭坐标轴
    
    # 调整布局
    plt.tight_layout()
    
    # 保存组合图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined image saved to: {output_path}")
    
    # 显示图片
    plt.show()

if __name__ == "__main__":
    # 输入图片路径
    combat_image_paths = [
        "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat/selfplay/combat_selfplay_normalized.png", # selfplay
        "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/vsbaseline/Combat_Vsbaseline_Normalized.png", # vs baseline
    ]
    
    reformation_image_paths = [
        "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/wedge_different_num_of_agents/Reformation Task (Wedge) Training Performance with Different Agent Numbers.png", # wedge different num of agents
        "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/reformation/agent_5_different_type_of_formation/Reformation Task (5 Agents) Training Performance with Different Formation Types.png" # 5 agents different type of formation
    ]

    combat_selfplay_with_different_communication_distances_and_different_num_envs_image_paths = [
        "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/combat_selfplay_new/combat_selfplay_different_distance_normalized.png", # combat selfplay with different communication distances
        "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/different NUM ENVS/Combat_Different_NUM_ENVS.png" # combat selfplay with different num envs
    ]
    
    # 定义输出路径
    combat_output_path = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combined_combat_plots.png"
    reformation_output_path = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combined_reformation_plots.png"
    combat_selfplay_with_different_communication_distances_and_different_num_envs_output_path = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combined_combat_selfplay_with_different_communication_distances_and_different_num_envs_plots.png"
    
    # 组合图片
    combine_images(combat_image_paths, combat_output_path, nrows=1, ncols=2, figsize=(16, 5))
    combine_images(reformation_image_paths, reformation_output_path, nrows=1, ncols=2, figsize=(16, 5))
    combine_images(combat_selfplay_with_different_communication_distances_and_different_num_envs_image_paths, combat_selfplay_with_different_communication_distances_and_different_num_envs_output_path, nrows=1, ncols=2, figsize=(16, 5))