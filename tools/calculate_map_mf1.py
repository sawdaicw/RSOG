import os
import numpy as np
from collections import defaultdict
import csv
from pathlib import Path

def parse_label_file(label_path):
    """解析标签文件，返回格式为[class_id, x_center, y_center, width, height]的列表"""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # 至少包含class_id和4个坐标
                    try:
                        class_id = int(parts[0])
                        box = list(map(float, parts[1:5]))
                        boxes.append([class_id] + box)
                    except ValueError:
                        print(f"Skipping malformed line in {label_path}: {line}")
    return boxes

def calculate_iou(box1, box2):
    """计算两个边界框的IoU (YOLO format: x_center, y_center, width, height)"""
    # 转换为 [x1, y1, x2, y2] (角坐标)
    box1_x1, box1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    box1_x2, box1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    
    box2_x1, box2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    box2_x2, box2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    # 计算交集区域
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算并集区域
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou

def calculate_ap(recalls, precisions):
    """计算AP (11点插值法)"""
    interp_precisions = []
    for t in np.arange(0, 1.1, 0.1):
        mask = recalls >= t
        if np.any(mask):
            interp_precisions.append(np.max(precisions[mask]))
        else:
            interp_precisions.append(0)
    return np.mean(interp_precisions)

def evaluate_dataset(image_paths, class_ids, pred_dir):
    """评估指定数据集和类别（仅计算IoU≥0.5的匹配）"""
    gt_boxes = []
    pred_boxes = []

    # 初始化统计变量
    total_gt = 0  # 所有类别的GT总数
    total_matched = 0  # 所有类别的成功匹配数（TP）

    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        img_filename = img_path.name
        label_filename = img_filename.replace('.jpg', '.txt')
        
        # 1. 获取 Ground Truth (GT)
        label_dir_name = img_path.parent.name.replace('images', 'labels')
        label_path = img_path.parent.parent / label_dir_name / label_filename
        
        # 针对 VisDrone 和 xView 不同的目录结构
        if 'VisDrone' in img_path_str:
             label_path = img_path.parent.parent / 'labels' / label_filename
        elif 'xView' in img_path_str:
             label_path = img_path.parent.parent / 'labels_yolo' / label_filename 

        # 确保路径正确 (回退到旧逻辑)
        if not os.path.exists(label_path):
             label_path_old = img_path_str.replace('images', 'labels').replace('.jpg', '.txt')
             if os.path.exists(label_path_old):
                 label_path = label_path_old

        gt_boxes.append(parse_label_file(label_path))
        
        # 2. 获取 Predictions (Pred)
        pred_path = pred_dir / label_filename
        pred_boxes.append(parse_label_file(pred_path))
    
    class_results = {class_id: {'tp': [], 'fp': [], 'scores': [], 'n_gt': 0, 'ious': []} 
                     for class_id in class_ids}
    
    for gt_img, pred_img in zip(gt_boxes, pred_boxes):
        gt_by_class = defaultdict(list)
        for gt in gt_img:
            if gt[0] in class_ids:
                gt_by_class[gt[0]].append(gt[1:])
        
        pred_by_class = defaultdict(list)
        for pred in pred_img:
            if pred[0] in class_ids:
                pred_by_class[pred[0]].append({'box': pred[1:], 'score': 1.0})
        
        for class_id in class_ids:
            class_gt = gt_by_class.get(class_id, [])
            class_pred = pred_by_class.get(class_id, [])
            class_results[class_id]['n_gt'] += len(class_gt)
            total_gt += len(class_gt)
            
            if not class_pred:
                continue
                
            class_pred_sorted = sorted(class_pred, key=lambda x: x['score'], reverse=True)
            gt_matched = [False] * len(class_gt)
            
            for pred in class_pred_sorted:
                pred_box = pred['box']
                max_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(class_gt):
                    if not gt_matched[gt_idx]:
                        iou = calculate_iou(pred_box, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                            best_gt_idx = gt_idx
                
                if max_iou >= 0.5:
                    class_results[class_id]['ious'].append(max_iou)
                
                if max_iou >= 0.5 and best_gt_idx != -1:
                    if not gt_matched[best_gt_idx]:
                        gt_matched[best_gt_idx] = True
                        class_results[class_id]['tp'].append(1)
                        class_results[class_id]['fp'].append(0)
                        total_matched += 1
                    else:
                        class_results[class_id]['tp'].append(0)
                        class_results[class_id]['fp'].append(1)
                else:
                    class_results[class_id]['tp'].append(0)
                    class_results[class_id]['fp'].append(1)
                
                class_results[class_id]['scores'].append(pred['score'])
    
    aps = []
    f1s = []
    ious = []
    
    for class_id in class_ids:
        tp = np.array(class_results[class_id]['tp'])
        fp = np.array(class_results[class_id]['fp'])
        scores = np.array(class_results[class_id]['scores'])
        n_gt = class_results[class_id]['n_gt']
        class_ious = class_results[class_id]['ious']
        
        if len(tp) == 0:
            aps.append(0)
            f1s.append(0)
            ious.append(0)
            continue
            
        sort_idx = np.argsort(-scores)
        tp = tp[sort_idx]
        fp = fp[sort_idx]
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / max(1, n_gt)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
        
        ap = calculate_ap(recalls, precisions)
        aps.append(ap)
        
        if len(precisions) > 0 and len(recalls) > 0:
            final_precision = precisions[-1]
            final_recall = recalls[-1]
            f1 = 2 * final_precision * final_recall / max(final_precision + final_recall, 1e-6)
        else:
            f1 = 0
        f1s.append(f1)
        
        mean_iou = np.mean(class_ious) if class_ious else 0
        ious.append(mean_iou)
    
    map_nc = np.mean(aps)
    mf1 = np.mean(f1s)
    miou = np.mean(ious)
    
    return map_nc, mf1, miou

def main():
    # 确定项目根目录
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # 结果根目录
    eval_root = project_root / 'results/eval'
    labels_root = eval_root / 'labels'
    coco_labels_root = eval_root / 'coco_labels'
    
    # 定义要评测的模型完整路径（相对于labels或coco_labels目录）
    model_paths = [
        # 3B Baseline
        'Qwen/Qwen2.5-VL-3B-Instruct/fixed/default',
        'Qwen/Qwen2.5-VL-3B-Instruct/open/mapping1',
        'Qwen/Qwen2.5-VL-3B-Instruct/open/mapping2',
        'Qwen/Qwen2.5-VL-3B-Instruct/open_ended/full_mapping',
        
        # 7B Baseline
        'Qwen/Qwen2.5-VL-7B-Instruct/fixed/default',
        'Qwen/Qwen2.5-VL-7B-Instruct/open/mapping1',
        'Qwen/Qwen2.5-VL-7B-Instruct/open/mapping2',
        'Qwen/Qwen2.5-VL-7B-Instruct/open_ended/full_mapping',
        
        # Falcon
        'Falcon/fixed/default',
        'Falcon/open/mapping1',
        'Falcon/open/mapping2',
        
        # LLaVA
        'llava-hf/llava-v1.6-vicuna-7b-hf/fixed/default',
        'llava-hf/llava-v1.6-vicuna-7b-hf/open/mapping1',
        'llava-hf/llava-v1.6-vicuna-7b-hf/open/mapping2',
        'llava-hf/llava-v1.6-vicuna-7b-hf/open_ended/full_mapping',
        
        # LAE-DINO
        'lae-dino/fixed/default',
        'lae-dino/fixed/open/mapping1',
        'lae-dino/fixed/open/mapping2',
    ]
    
    # 自动发现新训练的模型
    discovered_models = []
    for root_dir in [labels_root, coco_labels_root]:
        if not root_dir.exists():
            continue
        
        # 查找checkpoint目录
        checkpoint_patterns = [
            'checkpoint-3021-merged',  # 3B Stage 1
            'checkpoint-3216-merged',  # 3B Stage 2
            'checkpoint-12084-merged', # 7B Stage 1
            'export_v5_11968',         # 7B old LoRA
        ]
        
        for pattern in checkpoint_patterns:
            for checkpoint_dir in root_dir.rglob(pattern):
                # 获取相对路径
                try:
                    rel_base = checkpoint_dir.relative_to(root_dir)
                    # 添加所有任务配置
                    tasks = ['fixed/default', 'open/mapping1', 'open/mapping2', 'open_ended/full_mapping']
                    for task in tasks:
                        task_path = rel_base / task
                        full_task_dir = root_dir / task_path
                        if full_task_dir.exists() and any(full_task_dir.iterdir()):
                            discovered_models.append({
                                'path': str(task_path),
                                'root': root_dir
                            })
                except ValueError:
                    continue
    
    # 构建完整的候选模型列表
    candidate_models = []
    
    # 添加预定义的模型
    for model_path in model_paths:
        for root_dir in [labels_root, coco_labels_root]:
            full_path = root_dir / model_path
            if full_path.exists() and any(full_path.iterdir()):
                candidate_models.append({
                    'name': model_path,
                    'path': full_path,
                    'root': root_dir.name
                })
                break
    
    # 添加自动发现的模型
    for model_info in discovered_models:
        candidate_models.append({
            'name': model_info['path'],
            'path': model_info['root'] / model_info['path'],
            'root': model_info['root'].name
        })
    
    if not candidate_models:
        print("错误: 未找到任何模型结果目录!")
        print(f"请检查 {labels_root} 和 {coco_labels_root}")
        return
    
    print(f"找到 {len(candidate_models)} 个模型配置待评测")
    for model in candidate_models:
        print(f"  - {model['name']} ({model['root']})")
    print()
    
    # 定义要计算的类别
    visdrone_classes = [3, 8, 5]
    xview_classes = [48]
    
    # 读取测试图片列表
    image_list_path = project_root / 'datasets/VLAD_Remote/test_image_list.txt'
    with open(image_list_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    # 分离VisDrone和xView图片路径
    visdrone_paths = [p for p in image_paths if 'VisDrone' in p]
    xview_paths = [p for p in image_paths if 'xView' in p]
    n_visdrone = len(visdrone_paths)
    n_xview = len(xview_paths)
    total_samples = n_visdrone + n_xview
    
    # 计算权重
    weight_visdrone = n_visdrone / max(1, total_samples)
    weight_xview = n_xview / max(1, total_samples)
    
    # 准备结果文件
    os.makedirs(eval_root, exist_ok=True)
    result_txt_file = eval_root / 'model_comparison_results.txt'
    result_csv_file = eval_root / 'model_comparison_results.csv'
    
    # CSV文件头
    csv_header = [
        'Model',
        'VisDrone_mAPnc', 'VisDrone_mF1', 'VisDrone_mIoU',
        'xView_mAPnc', 'xView_mF1', 'xView_mIoU',
        'Weighted_mAPnc', 'Weighted_mF1', 'Weighted_mIoU'
    ]
    
    print(f"开始评测... 结果将保存到 {eval_root}\n")

    with open(result_txt_file, 'w') as txt_f, open(result_csv_file, 'w', newline='') as csv_f:
        txt_f.write("Model Evaluation Results\n")
        txt_f.write("="*80 + "\n")
        txt_f.write(f"VisDrone样本数: {n_visdrone}, xView样本数: {n_xview}\n")
        txt_f.write(f"权重分配: VisDrone={weight_visdrone:.2f}, xView={weight_xview:.2f}\n")
        txt_f.write("="*80 + "\n\n")
        
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(csv_header)
        
        for model_info in candidate_models:
            model_name = model_info['name']
            pred_dir = model_info['path']
            
            txt_f.write(f"Evaluating model: {model_name}\n")
            txt_f.write(f"Path: {pred_dir}\n")
            print(f"正在评测: {model_name}")
            print(f"路径: {pred_dir}")
            
            try:
                # 计算VisDrone指标
                visdrone_map, visdrone_mf1, visdrone_miou = evaluate_dataset(
                    visdrone_paths, visdrone_classes, pred_dir)
                
                # 计算xView指标
                xview_map, xview_mf1, xview_miou = evaluate_dataset(
                    xview_paths, xview_classes, pred_dir)
                
                # 按样本量加权平均
                avg_map = (visdrone_map * weight_visdrone + xview_map * weight_xview)
                avg_mf1 = (visdrone_mf1 * weight_visdrone + xview_mf1 * weight_xview)
                avg_miou = (visdrone_miou * weight_visdrone + xview_miou * weight_xview)
                
                # 写入文本结果
                txt_f.write(f"VisDrone Results (Classes: {visdrone_classes}):\n")
                txt_f.write(f"- mAPnc: {visdrone_map:.4f}\n")
                txt_f.write(f"- mF1: {visdrone_mf1:.4f}\n")
                txt_f.write(f"- mIoU: {visdrone_miou:.4f}\n")
                
                txt_f.write(f"\nxView Results (Classes: {xview_classes}):\n")
                txt_f.write(f"- mAPnc: {xview_map:.4f}\n")
                txt_f.write(f"- mF1: {xview_mf1:.4f}\n")
                txt_f.write(f"- mIoU: {xview_miou:.4f}\n")
                
                txt_f.write(f"\nWeighted Average (by sample size):\n")
                txt_f.write(f"- mAPnc: {avg_map:.4f}\n")
                txt_f.write(f"- mF1: {avg_mf1:.4f}\n")
                txt_f.write(f"- mIoU: {avg_miou:.4f}\n")
                
                txt_f.write("\n" + "="*80 + "\n\n")
                
                # 写入CSV行
                csv_row = [
                    model_name,
                    f"{visdrone_map:.4f}", f"{visdrone_mf1:.4f}", f"{visdrone_miou:.4f}",
                    f"{xview_map:.4f}", f"{xview_mf1:.4f}", f"{xview_miou:.4f}",
                    f"{avg_map:.4f}", f"{avg_mf1:.4f}", f"{avg_miou:.4f}"
                ]
                csv_writer.writerow(csv_row)
                
                print(f"  VisDrone: mAP={visdrone_map:.4f}, mF1={visdrone_mf1:.4f}, mIoU={visdrone_miou:.4f}")
                print(f"  xView: mAP={xview_map:.4f}, mF1={xview_mf1:.4f}, mIoU={xview_miou:.4f}")
                print(f"  Weighted: mAP={avg_map:.4f}, mF1={avg_mf1:.4f}, mIoU={avg_miou:.4f}\n")
                
            except Exception as e:
                error_msg = f"评测失败: {str(e)}\n"
                txt_f.write(error_msg)
                txt_f.write("="*80 + "\n\n")
                print(f"  错误: {str(e)}\n")
                continue
    
    print(f"\n评测完成! 结果已保存:")
    print(f"- 文本文件: {result_txt_file}")
    print(f"- CSV文件: {result_csv_file}")

if __name__ == '__main__':
    main()