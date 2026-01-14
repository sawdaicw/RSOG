import os
import json
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import torch
from transformers import BertTokenizer, BertModel

# 设置 Hugging Face 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class BertEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        print(f"🔄 正在加载 BERT 模型 (Device: {device})...")
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.model.eval()
        
        # 缓存，避免重复计算相同单词的 Embedding
        self.embedding_cache = {}

    def get_embedding(self, text):
        """获取文本的 BERT Embedding，带缓存"""
        # 预处理：转小写，去标点
        text = str(text).lower().strip().replace('.', '').replace(',', '')
        
        if not text:
            return None
            
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # 使用 [CLS] token
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"BERT Error for '{text}': {e}")
            return None

    def calculate_similarity(self, text1, text2):
        """计算余弦相似度"""
        if text1 == text2: return 1.0 # 完全匹配
        
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
            
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

def calculate_metrics(result_dir, similarity_threshold=0.85):
    """
    计算 Precision, Recall, F1
    :param similarity_threshold: BERT 相似度阈值，大于此值视为匹配成功
    """
    evaluator = BertEvaluator()
    
    json_files = glob.glob(os.path.join(result_dir, "**/*.json"), recursive=True)
    print(f"📂 找到 {len(json_files)} 个结果文件。")

    # 统计数据
    # TP: 预测对了 (相似度 > 阈值)
    # FP: 预测了但 GT 里没有 (或者相似度都不够)
    # FN: GT 里有但没预测出来
    stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    global_tp, global_fp, global_fn = 0, 0, 0
    valid_count = 0

    for json_file in tqdm(json_files, desc="Calculating Metrics"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except:
            continue

        # 1. 获取 Ground Truth
        gt_raw = data.get('gt_object', [])
        # 清洗 GT 列表
        gt_objects = set()
        for obj in gt_raw:
            clean_obj = str(obj).lower().strip().replace("['", "").replace("']", "").replace("'", "")
            if clean_obj and clean_obj != "unknown" and clean_obj != "none":
                gt_objects.add(clean_obj)
        
        if not gt_objects:
            continue # 没有 GT 的图片跳过

        # 2. 获取预测结果
        detections = data.get('detections', [])
        pred_objects = set()
        if isinstance(detections, list):
            for det in detections:
                if isinstance(det, dict) and 'label' in det:
                    clean_label = str(det['label']).lower().strip()
                    pred_objects.add(clean_label)
        
        valid_count += 1

        # 3. 匹配逻辑 (基于 BERT 相似度的二分图匹配简化版)
        # 我们需要看 GT 中的每一个词，是否在 Pred 中找到了“语义相似”的词
        
        # --- 计算 Recall (针对每个 GT 找匹配) ---
        for gt in gt_objects:
            # 在预测列表中找最相似的一个
            best_sim = 0.0
            best_match = None
            
            # 先尝试精确匹配
            if gt in pred_objects:
                best_sim = 1.0
                best_match = gt
            else:
                # 否则跑 BERT
                for pred in pred_objects:
                    sim = evaluator.calculate_similarity(gt, pred)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = pred
            
            # 判定
            if best_sim >= similarity_threshold:
                stats[gt]['tp'] += 1 # 对于这个类别，算 TP
                global_tp += 1
            else:
                stats[gt]['fn'] += 1 # 没找到相似的，算 FN
                global_fn += 1

        # --- 计算 Precision (针对每个 Pred 找匹配) ---
        # 注意：这里简化处理。如果一个 Pred 匹配到了任意一个 GT，就算 TP (上面已经加过了)，否则算 FP。
        # 为了避免重复计算 TP，我们只计算 FP。
        
        for pred in pred_objects:
            # 在 GT 列表中找最相似的一个
            best_sim = 0.0
            
            if pred in gt_objects:
                best_sim = 1.0
            else:
                for gt in gt_objects:
                    sim = evaluator.calculate_similarity(pred, gt)
                    if sim > best_sim:
                        best_sim = sim
            
            # 如果最大的相似度都小于阈值，说明预测了一个完全不相关的东西 -> FP
            if best_sim < similarity_threshold:
                # 归类到 pred 自己的名字下
                stats[pred]['fp'] += 1
                global_fp += 1
            # 如果 >= 阈值，上面 Recall 阶段已经算过 TP 了，这里不重复加 TP

    # --- 输出结果 ---
    results_list = []
    for cls, metrics in stats.items():
        tp = metrics['tp']
        fp = metrics['fp']
        fn = metrics['fn']
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        
        results_list.append({
            "Class": cls,
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1": round(f1, 4),
            "Support (TP+FN)": tp + fn
        })

    df = pd.DataFrame(results_list)
    if not df.empty:
        df = df.sort_values(by="Support (TP+FN)", ascending=False)

    # 全局指标
    micro_p = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    micro_r = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    print("\n" + "="*60)
    print(f"📊 基于 BERT 语义相似度的评估结果 (Threshold={similarity_threshold})")
    print("="*60)
    print(f"Global Precision : {micro_p:.4f}")
    print(f"Global Recall    : {micro_r:.4f}")
    print(f"Global F1-Score  : {micro_f1:.4f}")
    print("="*60)
    
    if not df.empty:
        print(df.head(20).to_string(index=False))
        output_csv = os.path.join(result_dir, "bert_metrics.csv")
        df.to_csv(output_csv, index=False)
        print(f"\n💾 结果已保存: {output_csv}")

if __name__ == "__main__":
    import argparse
    
    # 默认路径（你刚才指定的路径）
    DEFAULT_DIR = "/home/zirui/.cursor-server/Qwen2.5-VL-FT-Remote/results/eval_qwen_instruction/labels/home/zirui/.cursor-server/Qwen2.5-VL-FT-Remote/export_v5_11968/open_ended"

    parser = argparse.ArgumentParser(description='计算 BERT 语义相似度指标')
    parser.add_argument('--dir', type=str, default=DEFAULT_DIR, help='包含 JSON 结果文件的目录路径')
    parser.add_argument('--threshold', type=float, default=0.85, help='BERT 相似度阈值 (0-1)')
    
    args = parser.parse_args()
    
    if os.path.exists(args.dir):
        print(f"🎯 正在评估目录: {args.dir}")
        calculate_metrics(args.dir, similarity_threshold=args.threshold)
    else:
        print(f"❌ 路径不存在: {args.dir}")
        print("提示：请检查路径是否正确，或通过 --dir 参数指定。")