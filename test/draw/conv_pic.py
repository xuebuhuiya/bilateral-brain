import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# 定义数据
data = {
    "W": [0.668278, 0.67304, 0.638639, 0.679267, 0.618469, 0.615876, 0.67587, 0.68025, 0.67493],
    "X": [0.733215, 0.77052, 0.753339, 0.775459, 0.737378, 0.733759, 0.77552, 0.77818, 0.77232],
    "AA": [0.003802914, 0.004144394, 0.003830563, 0.005895888, 0.005253216, 0.006789572, 0.005239179, 0.004272717, 0.002991488],
    "AB": [0.004526591, 0.003377968, 0.003389213, 0.004397035, 0.005553311, 0.006826788, 0.004339176, 0.004200476, 0.012215546],
}

# 创建DataFrame
df = pd.DataFrame(data)

# 准备绘图数据
labels = ['single\nhemisphere', 'Bilateral with \nspecialization', 'Concatenation \nwith 1x1\nConvolution', 'Fixed-Weight\nAddition', 'Half-Feature\nWeighted\nAddition', 'Pooling \n and \nSummation', 'Attention\nWeighted\nAddition', 'Gated Feature\nTransmission', 'Gated Inter\ncommunication\n Units']
fine_ave = df['W']
coarse_ave = df['X']
fine_error = df['AA']
coarse_error = df['AB']

x_positions = np.arange(len(labels))  # 标签位置

# 定义柱子的宽度和间距
bar_width = 0.3  # 每个柱子的宽度
bar_gap = 0.05   # 柱子之间的间距

# 计算每个柱子的X位置
x_fine = x_positions - (bar_width + bar_gap)/2
x_coarse = x_positions + (bar_width + bar_gap)/2

# 创建绘图
fig, ax = plt.subplots(figsize=(10, 6))

# 设置Y轴范围
ax.set_ylim([0.5, 0.85]) 
ax.set_ylabel('Values (%)')

# 调整Y轴为百分比格式
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

# 启用水平网格线
ax.grid(axis='y', linestyle='-', linewidth=0.7)
ax.set_axisbelow(True)

# 设置x轴标签
ax.set_xticks(x_positions)
ax.set_xticklabels(labels)

# 定义绘制垂直渐变柱子的函数
def gradient_bar(ax, x_center, y_bottom, width, height, color_top, color_bottom):
    N = 256  # 渐变的分辨率
    # 创建垂直方向的梯度数组
    gradient = np.linspace(0, 1, N).reshape(N, 1)
    # 创建线性渐变的颜色映射
    cmap = LinearSegmentedColormap.from_list('gradient', [color_bottom, color_top])
    # 设置渐变的绘制范围
    extent = (x_center - width/2, x_center + width/2, y_bottom, y_bottom + height)
    # 在柱子的位置绘制渐变图像
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, zorder=2, alpha=1, clip_on=True)

# 绘制带有垂直渐变的柱子和误差线
for i in range(len(x_positions)):
    # 渐变起始位置
    y_bottom = 0.3  # 从Y=0.3开始
    # Fine Ave 柱子
    fine_height = fine_ave[i] - y_bottom  # 高度从0.3到实际值
    # 绘制Fine Ave的渐变柱子
    gradient_bar(ax, x_fine[i], y_bottom, bar_width, fine_height, 'blue', 'white')
    # 绘制误差线
    ax.errorbar(x_fine[i], fine_ave[i], yerr=fine_error[i], fmt='none', ecolor='black', capsize=5, zorder=3)
    
    # Coarse Ave 柱子
    coarse_height = coarse_ave[i] - y_bottom
    # 绘制Coarse Ave的渐变柱子
    gradient_bar(ax, x_coarse[i], y_bottom, bar_width, coarse_height, 'orange', 'white')
    # 绘制误差线
    ax.errorbar(x_coarse[i], coarse_ave[i], yerr=coarse_error[i], fmt='none', ecolor='black', capsize=5, zorder=3)

# 手动添加图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', edgecolor='blue', label='Fine class'),
                   Patch(facecolor='orange', edgecolor='orange', label='Coarse class')]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=False, shadow=False, ncol=2, frameon=False)

# 设置标题
ax.set_title('Connecting features after convolution blocks')

# 设置X轴范围，添加左右空隙
ax.set_xlim([x_positions[0] - 0.5, x_positions[-1] + 0.5])

# 确保布局正确
plt.tight_layout()

# 显示图形
plt.show()
