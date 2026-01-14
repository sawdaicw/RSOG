import numpy as np
import matplotlib.pyplot as plt

# 数据
categories = ['FT', 'Qwen2.5-VL-7B', 'LLaVA', 'Falcon', 'lae-dino']
metrics = ['mAPnc', 'mF1', 'mIoU']

# fixed 数据
fixed_data = {
    'mAPnc': [0.1058, 0.107, 0.0237, 0.0988, 0.1403],
    'mF1': [0.2333, 0.1507, 0.0078, 0.1725, 0.2677],
    'mIoU': [0.7073, 0.7323, 0.605, 0.5272, 0.7418]
}

# open 数据 (注意: lae-dino没有open数据)
open_data = {
    'mAPnc': [0.05535, 0.02595, 0.01155, 0.0633, np.nan],
    'mF1': [0.13855, 0.07455, 0.0046, 0.0206, np.nan],
    'mIoU': [0.5765, 0.5825, 0.5098, 0.66855, np.nan]
}

# open-ended 数据 (注意: Falcon和lae-dino没有open-ended数据)
open_ended_data = {
    'mAPnc': [0.0365, 0.05, 0.0347, np.nan, np.nan],
    'mF1': [0.0256, 0.0155, 0.0015, np.nan, np.nan],
    'mIoU': [0.7199, 0.756, 0.5868, np.nan, np.nan]
}

# 创建图表
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
plt.subplots_adjust(hspace=0.5)

# 设置柱状图的宽度和位置
bar_width = 0.25
index = np.arange(len(categories))

# 绘制每个指标的图表
for i, metric in enumerate(metrics):
    ax = axes[i]
    
    # 绘制fixed数据
    bars1 = ax.bar(index - bar_width, fixed_data[metric], bar_width, label='Fixed', color='b')
    
    # 绘制open数据
    bars2 = ax.bar(index, open_data[metric], bar_width, label='Open', color='g')
    
    # 绘制open-ended数据
    bars3 = ax.bar(index + bar_width, open_ended_data[metric], bar_width, label='Open-ended', color='r')
    
    ax.set_title(metric)
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_xticks(index)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # 添加数据标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=8)

plt.suptitle('Performance Comparison Across Different Settings', y=0.98, fontsize=14)

# 保存图表为PNG文件（也可以改为PDF或SVG格式）
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
print("图表已保存为 performance_comparison.png")

plt.show()