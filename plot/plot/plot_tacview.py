import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from PIL import Image
import numpy as np

def combine_images(image_paths, titles, output_path, main_title):
    # 创建图形和网格
    fig = plt.figure(figsize=(18, 7), dpi=300)
    
    # 创建网格，确保每列宽度相同
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[0.1, 1])
    
    # 添加总标题（作为一个跨越所有列的文本框）
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.text(0.5, 0.5, main_title, fontsize=14, ha='center', va='center')
    title_ax.axis('off')  # 隐藏坐标轴
    
    # 添加每个图片
    for i, (img_path, title) in enumerate(zip(image_paths, titles)):
        # 确保图片存在
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}")
            continue
            
        ax = fig.add_subplot(gs[1, i])
        
        # 使用PIL打开图片以获取正确的尺寸比例
        pil_img = Image.open(img_path)
        img = np.array(pil_img)
        
        # 在相同的轴空间中显示图片
        ax.imshow(img)
        ax.set_title(title, fontsize=14, pad=10)
        ax.axis('off')  # 隐藏坐标轴
        
        # 保持纵横比但固定大小
        ax.set_aspect('auto')
    
    # 调整布局
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存图片
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Combined image saved to {output_path}")
    plt.close()

# 图片路径
image_paths = [
    "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/tacview/combat_selfplay_hierarchy_agent_4_seed42.png",
    "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/tacview/combat_agent_100_selfplay_hierarchy.png",
    "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/tacview/formation_19_diamond.png"
]

# 标题
titles = [
    "2v2 Hierarchical SelfPlay",
    "50v50 Hierarchical SelfPlay",
    "Wedge Reformation(19 agents)"
]

# 总标题
main_title = "Render Result in Tacview"

# 输出路径
output_path = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/tacview/combined_tacview.png"

# 执行合并
combine_images(image_paths, titles, output_path, main_title)