import os
import numpy as np
from collections import defaultdict
from pathlib import Path

# ================= 配置区域 =================
# 1. 修正路径：指向 full_mapping 文件夹
PRED_ROOT_DIR = "./results/eval/coco_labels/eval_qwen_instruction/open_ended/full_mapping"

# 2. 图片列表文件路径
IMAGE_LIST_PATH = "./datasets/VLAD_Remote/test_image_list.txt"

# 3. 评测的类别 ID
VISDRONE_CLASSES = [3, 5, 8] 
XVIEW_CLASSES = [48]
# ===========================================

def parse_label_file(label_path):
    """解析标签文件"""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(float(parts[0]))
                        box = list(map(float, parts[1:5]))
                        score = float(parts[5]) if len(parts) > 5 else 1.0
                        boxes.append({'class_id': class_id, 'box': box, 'score': score})
                    except ValueError:
                        continue
    return boxes

def parse_gt_file(label_path):
    """解析GT文件"""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(float(parts[0]))
                        box = list(map(float, parts[1:5]))
                        boxes.append([class_id] + box)
                    except ValueError:
                        continue
    return boxes

def calculate_iou(box1, box2):
    """计算 IoU"""
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    x_left = max(b1_x1, b2_x1)
    y_top = max(b1_y1, b2_y1)
    x_right = min(b1_x2, b2_x2)
    y_bottom = min(b1_y2, b2_y2)

    if x_right < x_left or y_bottom < y_top: return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    return inter / (area1 + area2 - inter + 1e-6)

def calculate_ap(recalls, precisions):
    """计算 AP"""
    if len(recalls) == 0: return 0.0
    interp_precisions = []
    for t in np.arange(0, 1.1, 0.1):
        mask = recalls >= t
        val = np.max(precisions[mask]) if np.any(mask) else 0.0
        interp_precisions.append(val)
    return np.mean(interp_precisions)

def evaluate_subset(image_paths, class_ids, subset_name):
    """评估子集"""
    # 修正：直接使用 PRED_ROOT_DIR，不再拼接子文件夹名
    pred_dir = Path(PRED_ROOT_DIR)
    
    if not pred_dir.exists():
        print(f"⚠️ 严重警告: 预测目录不存在 {pred_dir}")
        print("请检查 json_to_coco.py 是否成功生成了 txt 文件。")
        return 0, 0, 0

    gt_data = []
    pred_data = []
    
    print(f"🔄 正在分析 {subset_name} 数据 (共 {len(image_paths)} 张)...")

    missing_pred_count = 0

    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        txt_name = img_path.stem + '.txt'
        
        # 1. 寻找 GT
        label_filename = txt_name
        possible_gt_paths = [
            img_path.parent.parent / 'labels' / label_filename,
            img_path.parent.parent / 'labels_yolo' / label_filename,
            Path(str(img_path.parent).replace('images', 'labels')) / label_filename
        ]
        
        gt_path = None
        for p in possible_gt_paths:
            if p.exists():
                gt_path = p
                break
        
        if gt_path:
            gt_data.append(parse_gt_file(gt_path))
        else:
            gt_data.append([])

        # 2. 寻找 Prediction (都在 full_mapping 目录下)
        pred_file = pred_dir / txt_name
        if pred_file.exists():
            pred_data.append(parse_label_file(pred_file))
        else:
            pred_data.append([])
            missing_pred_count += 1
            
    if missing_pred_count > 0:
        print(f"   (提示: 有 {missing_pred_count} 张图片没有找到预测结果文件，可能被判定为无目标)")

    # --- 计算指标逻辑 ---
    class_metrics = {cid: {'tp': [], 'fp': [], 'scores': [], 'n_gt': 0, 'ious': []} for cid in class_ids}

    for gt_list, pred_list in zip(gt_data, pred_data):
        gt_map = defaultdict(list)
        for item in gt_list:
            if item[0] in class_ids:
                gt_map[item[0]].append(item[1:])
        
        pred_map = defaultdict(list)
        for item in pred_list:
            if item['class_id'] in class_ids:
                pred_map[item['class_id']].append(item)

        for cid in class_ids:
            gts = gt_map[cid]
            preds = pred_map[cid]
            class_metrics[cid]['n_gt'] += len(gts)
            
            if not preds: continue
            
            preds.sort(key=lambda x: x['score'], reverse=True)
            gt_matched = [False] * len(gts)
            
            for p in preds:
                p_box = p['box']
                best_iou = 0
                best_idx = -1
                
                for idx, g_box in enumerate(gts):
                    if not gt_matched[idx]:
                        iou = calculate_iou(p_box, g_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx
                
                if best_iou >= 0.5:
                    class_metrics[cid]['ious'].append(best_iou)
                    class_metrics[cid]['tp'].append(1)
                    class_metrics[cid]['fp'].append(0)
                    gt_matched[best_idx] = True
                else:
                    class_metrics[cid]['tp'].append(0)
                    class_metrics[cid]['fp'].append(1)
                
                class_metrics[cid]['scores'].append(p['score'])

    aps, f1s, ious = [], [], []
    for cid in class_ids:
        res = class_metrics[cid]
        if not res['scores']:
            if res['n_gt'] > 0: aps.append(0); f1s.append(0); ious.append(0)
            continue

        tp = np.array(res['tp'])
        fp = np.array(res['fp'])
        scores = np.array(res['scores'])
        n_gt = res['n_gt']
        
        sort_idx = np.argsort(-scores)
        tp = tp[sort_idx]
        fp = fp[sort_idx]
        
        recalls = np.cumsum(tp) / max(1, n_gt)
        precisions = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-6)
        
        aps.append(calculate_ap(recalls, precisions))
        
        f1 = 0
        if len(precisions) > 0:
            p, r = precisions[-1], recalls[-1]
            f1 = 2 * p * r / (p + r + 1e-6)
        f1s.append(f1)
        
        ious.append(np.mean(res['ious']) if res['ious'] else 0)

    return (np.mean(aps) if aps else 0.0), (np.mean(f1s) if f1s else 0.0), (np.mean(ious) if ious else 0.0)

def main():
    print(f"📊 评测数据源: {PRED_ROOT_DIR}")
    
    if not os.path.exists(IMAGE_LIST_PATH):
        print(f"❌ 找不到图片列表文件: {IMAGE_LIST_PATH}")
        return
        
    with open(IMAGE_LIST_PATH, 'r') as f:
        all_images = [line.strip() for line in f if line.strip()]

    visdrone_imgs = [x for x in all_images if 'VisDrone' in x]
    xview_imgs = [x for x in all_images if 'xView' in x]
    
    print(f"🖼️  图片总数: {len(all_images)} (VisDrone: {len(visdrone_imgs)}, xView: {len(xview_imgs)})")
    print("=" * 60)

    print(">>> 正在评测 VisDrone 数据集...")
    v_map, v_f1, v_iou = evaluate_subset(visdrone_imgs, VISDRONE_CLASSES, 'visdrone')
    
    print("\n>>> 正在评测 xView 数据集...")
    x_map, x_f1, x_iou = evaluate_subset(xview_imgs, XVIEW_CLASSES, 'xview')

    total = len(all_images)
    w_v = len(visdrone_imgs) / max(total, 1)
    w_x = len(xview_imgs) / max(total, 1)
    
    avg_map = v_map * w_v + x_map * w_x
    avg_f1 = v_f1 * w_v + x_f1 * w_x
    avg_iou = v_iou * w_v + x_iou * w_x

    print("\n" + "=" * 60)
    print("🏆 最终评测结果")
    print("=" * 60)
    print(f"VisDrone (mAP/F1/IoU):  {v_map:.4f} / {v_f1:.4f} / {v_iou:.4f}")
    print(f"xView    (mAP/F1/IoU):  {x_map:.4f} / {x_f1:.4f} / {x_iou:.4f}")
    print("-" * 60)
    print(f"Global Weighted Average: mAP={avg_map:.4f}, F1={avg_f1:.4f}, IoU={avg_iou:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()