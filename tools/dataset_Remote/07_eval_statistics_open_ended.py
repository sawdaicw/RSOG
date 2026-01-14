import json
import os
import glob
from PIL import Image
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# 1. 设置镜像，防止联网报错
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 2. 初始化 BERT 模型 (优先使用本地，否则联网)
print("🔄 正在初始化 BERT 模型...")
model_path = "./bert_local"  # 之前建议下载的本地路径
if not os.path.exists(model_path):
    print("⚠️ 未找到本地 bert_local 文件夹，尝试从 Hugging Face 镜像在线加载...")
    model_path = 'bert-base-uncased'

try:
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    model.eval()
    print("✅ BERT 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("建议先运行 tools/download_bert.py 下载模型到本地。")
    exit(1)

def get_bert_embedding(text):
    """使用BERT获取文本的嵌入向量"""
    try:
        if not text.strip():
            return None
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
        return embedding
    except Exception as e:
        print(f"获取BERT嵌入时出错: {e}")
        return None

def calculate_bert_similarity(text1, text2):
    """使用BERT嵌入计算两个文本之间的余弦相似度"""
    try:
        def clean_text(text):
            if isinstance(text, list):
                text = ' '.join(text)
            text = str(text).lower().replace('[', '').replace(']', '').replace("'", "").replace(".", "")
            return text
            
        text1_clean = clean_text(text1)
        text2_clean = clean_text(text2)
        
        if not text1_clean.strip() or not text2_clean.strip():
            return 0.0
            
        embedding1 = get_bert_embedding(text1_clean)
        embedding2 = get_bert_embedding(text2_clean)
        
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float(similarity)
    except Exception as e:
        print(f"计算BERT相似度时出错: {e}")
        return 0.0

def get_image_size(image_path):
    """获取图像尺寸"""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"无法获取图像尺寸: {image_path}, 错误: {e}")
        return None

def get_class_mappings():
    return {
        'full_mapping': {
            'xview': {
                'Fixed-wing Aircraft': 0, 'Small Aircraft': 1, 'Cargo Plane': 2, 'Helicopter': 3, 'Passenger Vehicle': 4,
                'Small Car': 5, 'Bus': 6, 'Pickup Truck': 7, 'Utility Truck': 8, 'Truck': 9, 'Cargo Truck': 10,
                'Truck w/Box': 11, 'Truck Tractor': 12, 'Trailer': 13, 'Truck w/Flatbed': 14, 'Truck w/Liquid': 15,
                'Crane Truck': 16, 'Railway Vehicle': 17, 'Passenger Car': 18, 'Cargo Car': 19, 'Flat Car': 20,
                'Tank car': 21, 'Locomotive': 22, 'Maritime Vessel': 23, 'Motorboat': 24, 'Sailboat': 25, 'Tugboat': 26,
                'Barge': 27, 'Fishing Vessel': 28, 'Ferry': 29, 'Yacht': 30, 'Container Ship': 31, 'Oil Tanker': 32,
                'Engineering Vehicle': 33, 'Tower crane': 34, 'Container Crane': 35, 'Reach Stacker': 36,
                'Straddle Carrier': 37, 'Mobile Crane': 38, 'Dump Truck': 39, 'Haul Truck': 40, 'Scraper/Tractor': 41,
                'Front loader/Bulldozer': 42, 'Excavator': 43, 'Cement Mixer': 44, 'Ground Grader': 45, 'Hut/Tent': 46,
                'Shed': 47, 'Building': 48, 'Aircraft Hangar': 49, 'Damaged Building': 50, 'Facility': 51,
                'Construction Site': 52, 'Vehicle Lot': 53, 'Helipad': 54, 'Storage Tank': 55, 'Shipping container lot': 56,
                'Shipping Container': 57, 'Pylon': 58, 'Tower': 59
            },
            'visdrone': {
                'pedestrian': 0, 'people': 1, 'bicycle': 2, 'car': 3, 'van': 4,
                'truck': 5, 'tricycle': 6, 'awning-tricycle': 7, 'bus': 8, 'motor': 9
            }
        }
    }

def calculate_similarity_to_gt_class(generated_text, gt_object):
    return calculate_bert_similarity(generated_text, gt_object)

def get_true_class_id(gt_object, class_mapping):
    try:
        def clean_text(text):
            if isinstance(text, list):
                text = ' '.join(text)
            text = str(text).lower().replace('[', '').replace(']', '').replace("'", "").replace(".", "")
            return text
        gt_clean = clean_text(gt_object)
        for class_name, class_id in class_mapping.items():
            if class_name.lower() in gt_clean or gt_clean in class_name.lower():
                return class_id
        return 0
    except Exception as e:
        print(f"获取真实类别ID时出错: {e}")
        return 0

def convert_json_to_coco_txt(json_file_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    all_mappings = get_class_mappings()
    
    json_files = glob.glob(os.path.join(json_file_path, '*.json'))
    print(f"📂 找到 {len(json_files)} 个 JSON 文件，准备处理...")

    count = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            image_path = data.get('image_path')
            if not image_path or not os.path.exists(image_path):
                # print(f"警告: 图像不存在: {image_path}")
                continue
                
            img_size = get_image_size(image_path)
            if img_size is None:
                continue
            img_width, img_height = img_size
            
            dataset_type = 'xview' if 'xview' in image_path.lower() else 'visdrone'
            
            for mapping_name, mappings in all_mappings.items():
                mapping_folder = os.path.join(output_folder, mapping_name)
                os.makedirs(mapping_folder, exist_ok=True)
                
                class_mapping = mappings.get(dataset_type, {})
                output_lines = []
                detections = data.get('detections', [])
                
                generated_object = data.get('object', '')
                gt_object = data.get('gt_object', '')
                similarity_score = calculate_similarity_to_gt_class(generated_object, gt_object)
                true_class_id = get_true_class_id(gt_object, class_mapping)
                
                if isinstance(detections, list):
                    for bbox in detections:
                        if len(bbox) != 4: continue
                        x1, y1, x2, y2 = bbox
                        
                        x_center = ((x1 + x2) / 2) / img_width
                        y_center = ((y1 + y2) / 2) / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        width = max(0.0, min(1.0, width))
                        height = max(0.0, min(1.0, height))
                        
                        output_lines.append(f"{true_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {similarity_score:.6f}")
                
                base_name = os.path.splitext(os.path.basename(json_file))[0]
                output_file = os.path.join(mapping_folder, f"{base_name}.txt")
                with open(output_file, 'w') as f:
                    f.write('\n'.join(output_lines))
                count += 1
        except Exception as e:
            print(f"处理文件 {json_file} 出错: {e}")

    print(f"✅ 处理完成！共生成了 {count} 个 TXT 文件。")

def main():

    json_folder = './results/eval_qwen_instruction/labels/Qwen/Qwen2.5-VL-7B-Instruct/open_ended'
    output_folder = './results/eval/coco_labels/eval_qwen7B_instruction/open_ended'
    print(f"📥 输入目录: {json_folder}")
    print(f"📤 输出目录: {output_folder}")
    
    if not os.path.exists(json_folder):
        print(f"❌ 错误: 输入目录不存在! 请检查路径是否正确。")
        return

    convert_json_to_coco_txt(json_folder, output_folder)

if __name__ == '__main__':
    main()