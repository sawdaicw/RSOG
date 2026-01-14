import matplotlib

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np

# ================= 核心配置区域 =================

# 1. 目标图片路径
TARGET_IMAGE_PATH = "./datasets/VLAD_Remote/VisDrone/VisDrone2019-DET-train/images/9999955_00000_d_0000064.jpg"

# 2. 结果根目录 (TXT 所在的父级目录)
RESULT_ROOT = "./results/eval/coco_labels"

# 3. 模型配置
MODELS_CONFIG = {
    "Ours_Fixed": {
        "name": "Ours (instr)",
        "path": "eval_qwen_instruction/open_ended/full_mapping",
        "color": "#d62728" # 红色
    },
    "Ours_Instr": {
        "name": "Ours (no instr)",
        "path": "ms-swift/output/export_v5_11968/open_ended/full_mapping", 
        "color": "#2ca02c" # 绿色
    }
}

# ===================================================

def parse_txt_boxes(txt_path, img_width, img_height):
    """解析 TXT (class xc yc w h) 转为像素坐标"""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
        
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    xc, yc, w, h = map(float, parts[1:5])
                    
                    x1 = (xc - w/2) * img_width
                    y1 = (yc - h/2) * img_height
                    w_pixel = w * img_width
                    h_pixel = h * img_height
                    
                    boxes.append([x1, y1, w_pixel, h_pixel])
                except ValueError:
                    continue
    return boxes

def main():
    if not os.path.exists(TARGET_IMAGE_PATH):
        print(f"❌ 错误: 找不到原始图片: {TARGET_IMAGE_PATH}")
        return

    # 1. 读取原始图片
    img_bgr = cv2.imread(TARGET_IMAGE_PATH)
    if img_bgr is None:
        print("❌ 错误: 图片读取失败")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img_rgb.shape

    # 2. 准备绘图
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    if not isinstance(axes, np.ndarray): axes = [axes] 
    axes = axes.flatten() 
    
    img_name = os.path.basename(TARGET_IMAGE_PATH)
    txt_name = os.path.splitext(img_name)[0] + ".txt"

    print(f"🎨 正在处理: {img_name}")
    
    # 3. 遍历绘制
    model_keys = ["Ours_Fixed", "Ours_Instr"]
    
    for idx, key in enumerate(model_keys):
        cfg = MODELS_CONFIG[key]
        ax = axes[idx]
        
        # 寻找结果 txt
        result_txt_path = os.path.join(RESULT_ROOT, cfg["path"], txt_name)
        boxes = parse_txt_boxes(result_txt_path, w_img, h_img)
        
        # 绘图
        ax.imshow(img_rgb)
        ax.set_title(f"{cfg['name']}\nDetected: {len(boxes)}", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # 画框
        for box in boxes:
            rect = patches.Rectangle(
                (box[0], box[1]), box[2], box[3], 
                linewidth=2, edgecolor=cfg["color"], facecolor='none'
            )
            ax.add_patch(rect)
        
        print(f"  - {cfg['name']}: {len(boxes)} 个目标")

    # 4. 直接保存在当前目录
    plt.tight_layout()
    
    # 保存文件名：result_原文件名.jpg
    save_name = f"result_{img_name}"
    save_path = os.path.join(os.getcwd(), save_name)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 图片已保存在当前目录下: \n👉 {save_path}")

if __name__ == "__main__":
    main()