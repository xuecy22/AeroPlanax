import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

def combine_images(image_paths, output_path, nrows=1, ncols=4, figsize=(20, 5)):
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
            print(f"警告: 只能显示{nrows*ncols}张图片，忽略剩余图片")
            break
            
        row = i // ncols
        col = i % ncols
        
        # 读取图片
        img = mpimg.imread(img_path)
        
        # 在指定位置添加子图
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.axis('off')  # 关闭坐标轴
        
        # 可选：添加图片标题
        # img_name = os.path.basename(img_path).split('.')[0]
        # ax.set_title(img_name, fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存组合图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"已保存组合图片到: {output_path}")
    
    # 显示图片
    plt.show()

if __name__ == "__main__":
    # 输入四个图片路径
    image_paths = [
        "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat/selfplay/combat_selfplay_single_plot.png",
        # 在这里添加其他三张图片的路径
        "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat/selfplay/combat_selfplay_single_plot.png",
        "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat/selfplay/combat_selfplay_single_plot.png",
        "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat/selfplay/combat_selfplay_single_plot.png"
    ]
    
    # 定义输出路径
    output_path = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat/combined_plots.png"
    
    # 组合图片
    combine_images(image_paths, output_path, nrows=1, ncols=4, figsize=(20, 5))