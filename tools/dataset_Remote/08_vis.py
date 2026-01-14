import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
import random
from pathlib import Path
import matplotlib.colors as mcolors
import numpy as np

def get_distinct_colors(n):
    """生成n种易于区分的颜色"""
    if n <= 10:
        return plt.cm.tab10.colors[:n]
    elif n <= 20:
        return plt.cm.tab20.colors[:n]
    else:
        # 使用HSV颜色空间生成更多颜色
        hues = np.linspace(0, 1, n, endpoint=False)
        return [mcolors.hsv_to_rgb((h, 0.8, 0.9)) for h in hues]

def visualize_bbox(json_path, output_dir):
    """可视化单个JSON文件的标注"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 读取图片
        image_path = data['image_path']
        if not os.path.exists(image_path):
            print(f"警告: 图片文件不存在: {image_path}")
            return False
            
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告: 无法加载图片: {image_path}")
            return False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 创建图形和轴
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)

        # 自动检测所有类别
        all_classes = set()
        for class_name, bboxes in data.get('detections', {}).items():
            if bboxes and isinstance(bboxes, list):  # 只包含有检测框的类别
                all_classes.add(class_name)
        
        if not all_classes:
            print(f"警告: 文件 {json_path} 中没有检测到任何类别")
            plt.close()
            return False

        # 为每个类别分配颜色
        class_colors = {}
        colors = get_distinct_colors(len(all_classes))
        for i, class_name in enumerate(sorted(all_classes)):
            class_colors[class_name] = colors[i]

        # 绘制每个检测框
        for class_name, bboxes in data.get('detections', {}).items():
            if not bboxes or not isinstance(bboxes, list):
                continue
            
            color = class_colors.get(class_name, 'white')
            
            for bbox in bboxes:
                if len(bbox) != 4:  # 确保bbox格式正确 [xmin, ymin, xmax, ymax]
                    continue
                    
                xmin, ymin, xmax, ymax = bbox
                width = xmax - xmin
                height = ymax - ymin
                
                rect = patches.Rectangle(
                    (xmin, ymin), width, height,
                    linewidth=2, edgecolor=color, facecolor='none',
                    label=class_name
                )
                ax.add_patch(rect)
                
                ax.text(
                    xmin, ymin-5, class_name,
                    color=color, fontsize=10, weight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
                )

        plt.title(f"Visualization of {Path(image_path).name}", fontsize=14)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:  # 只有有图例时才添加
            plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        plt.axis('off')
        plt.tight_layout()

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出路径
        output_filename = Path(json_path).stem + '.png'
        output_path = os.path.join(output_dir, output_filename)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"可视化结果已保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"处理文件 {json_path} 时出错: {str(e)}")
        return False

def find_all_json_files(root_dir):
    """递归查找所有JSON文件"""
    json_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json') and not file.startswith('.'):
                json_files.append(os.path.join(root, file))
    return json_files

def sample_and_visualize_recursive(root_dir, sample_ratio=0.1, output_base_dir='results/visualization'):
    """
    递归遍历子文件夹，按比例采样文件进行可视化
    
    参数:
        root_dir: 包含子文件夹的根目录
        sample_ratio: 采样比例 (0~1)
        output_base_dir: 输出目录基础路径
    """
    # 查找所有JSON文件
    all_json_files = find_all_json_files(root_dir)
    if not all_json_files:
        print(f"警告: {root_dir} 中没有找到JSON文件")
        return
    
    # 按文件名分组（不考虑目录）
    file_groups = {}
    for json_file in all_json_files:
        filename = os.path.basename(json_file)
        if filename not in file_groups:
            file_groups[filename] = []
        file_groups[filename].append(json_file)
    
    # 计算需要采样的数量
    sample_count = max(1, int(len(file_groups) * sample_ratio))
    sampled_filenames = random.sample(list(file_groups.keys()), sample_count)
    
    print(f"采样 {sample_count} 个文件 (共 {len(file_groups)} 个唯一文件名)")
    print(f"采样的文件: {sampled_filenames}")
    
    # 处理每个采样的文件
    total_success = 0
    for filename in sampled_filenames:
        print(f"\n处理文件: {filename}")
        file_group = file_groups[filename]
        
        for json_path in file_group:
            # 计算输出目录，保持原始目录结构
            rel_path = os.path.relpath(os.path.dirname(json_path), root_dir)
            output_dir = os.path.join(output_base_dir, rel_path)
            
            if visualize_bbox(json_path, output_dir):
                total_success += 1
    
    print(f"\n处理完成: {total_success}/{sample_count*len(file_groups)} 个文件成功可视化")

if __name__ == "__main__":
    # 设置根目录和采样比例
    root_directory = "./results/eval/labels"  # 替换为你的根目录
    sampling_ratio = 0.01  # 10%的采样比例
    
    # 执行采样和可视化
    sample_and_visualize_recursive(
        root_dir=root_directory,
        sample_ratio=sampling_ratio,
        output_base_dir='results/visualization'
    )