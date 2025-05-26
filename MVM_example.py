import random
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull  
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 分层生成架构
class MultiTeamGenerator:
    def __init__(self, 
                 num_teams=4, 
                 units_per_team=3, 
                 map_radius=20000,
                 inter_team_min_dist=5000,
                 intra_team_spacing=1000):
        self.num_teams = num_teams
        self.units_per_team = units_per_team
        self.map_radius = map_radius
        self.inter_team_min_dist = inter_team_min_dist
        self.intra_team_spacing = intra_team_spacing

    def generate(self):
        # Step1: 生成团队基准点
        team_centers = self._generate_team_centers()
        
        # Step2: 为每个团队生成单位
        all_units = []
        for center in team_centers:
            units = self._generate_team_units(center)
            all_units.extend(units)
            
        return all_units

    def _generate_team_centers(self):
        """生成满足最小间距的团队中心点"""
        centers = []
        while len(centers) < self.num_teams:
            new_center = (
                random.uniform(-self.map_radius, self.map_radius),
                random.uniform(-self.map_radius, self.map_radius),
                random.uniform(3000, 6000)  # 高度范围
            )
            if all(self._distance(new_center, exist) >= self.inter_team_min_dist 
                   for exist in centers):
                centers.append(new_center)
        return centers

    def _generate_team_units(self, center):
        """在团队中心周围生成作战单位"""
        units = []
        for _ in range(self.units_per_team):
            # 在中心周围生成扰动位置
            dx = random.uniform(-self.intra_team_spacing, self.intra_team_spacing)
            dy = random.uniform(-self.intra_team_spacing, self.intra_team_spacing)
            dz = random.uniform(-500, 500)
            units.append((
                center[0] + dx,
                center[1] + dy,
                center[2] + dz
            ))
        return units

    def _distance(self, pos1, pos2):
        """3D欧氏距离计算"""
        return math.sqrt(sum((a-b)**2 for a,b in zip(pos1, pos2)))
    
# 2. 战术队形模板
def generate_formation(center, formation_type="wedge", spacing=800):
    """生成特定战术队形
    Args:
        formation_type: wedge(楔形), line(横队), diamond(菱形)
    """
    formations = {
        "wedge": [
            (0, 0, 0),  # 长机
            (-spacing, spacing, 0),
            (spacing, spacing, 0)
        ],
        "line": [
            (-spacing, 0, 0),
            (0, 0, 0),
            (spacing, 0, 0)
        ],
        "diamond": [
            (0, 0, 0),
            (-spacing, spacing, 0),
            (spacing, spacing, 0),
            (0, 2*spacing, 0)
        ]
    }
    offset = formations.get(formation_type, formations["wedge"])
    
    return [
        (center[0]+dx, center[1]+dy, center[2]+dz)
        for dx, dy, dz in offset
    ]

# 3D多团队可视化
def plot_3d_battlefield(units, team_colors):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制每个团队
    for team_idx, positions in enumerate(units):
        color = team_colors[team_idx % len(team_colors)]
        x, y, z = zip(*positions)
        ax.scatter(x, y, z, c=color, s=50, label=f'Team {team_idx+1}')
        
        # 绘制团队连接线
        ax.plot(x, y, z, color=color, alpha=0.3)
    
    ax.set_xlabel('X轴 (米)')
    ax.set_ylabel('Y轴 (米)')
    ax.set_zlabel('高度 (米)')
    plt.title("多对多空战初始部署")
    plt.legend()
    plt.show()
    # plt.savefig('multi_team_3d.png', dpi=300)
    # plt.close()

def plot_2d_tactical(units, team_colors):
    plt.figure(figsize=(10, 10))
    
    # 绘制团队区域
    for team_idx, positions in enumerate(units):
        color = team_colors[team_idx % len(team_colors)]
        x, y = zip(*[(p[0], p[1]) for p in positions])
        
        # 绘制单位
        plt.scatter(x, y, c=color, s=80, marker='^', edgecolors='k')
        
        # 绘制团队凸包
        hull = ConvexHull(np.array(list(zip(x,y))))
        for simplex in hull.simplices:
            plt.plot(np.array(x)[simplex], np.array(y)[simplex], 
                    color=color, linestyle='--', alpha=0.4)
    
    plt.xlabel("东西方向 (米)")
    plt.ylabel("南北方向 (米)")
    plt.title("多团队战术部署俯视图")
    plt.grid(True)
    plt.show()
    # plt.savefig('tactical_overview.png', dpi=300)
    # plt.close()

if __name__ == "__main__":
    # 参数配置
    config = {
        "num_teams": 4,          # 总团队数量
        "units_per_team": 5,     # 每队单位数
        "map_radius": 30000,     # 战场半径（米）
        "inter_team_min_dist": 8000,  # 团队间最小距离
        "intra_team_spacing": 1500    # 队内单位间距
    }
    
    # 生成部署
    generator = MultiTeamGenerator(**config)
    all_units = generator.generate()
    
    # 按团队分组
    team_units = [
        all_units[i*config["units_per_team"] : (i+1)*config["units_per_team"]] 
        for i in range(config["num_teams"])
    ]
    
    # 可视化
    colors = ['#FF3333', '#3333FF', '#33FF33', '#FF33FF']
    plot_3d_battlefield(team_units, colors)
    plot_2d_tactical(team_units, colors)