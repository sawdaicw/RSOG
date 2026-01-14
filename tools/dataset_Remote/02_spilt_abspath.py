import json
import random
import os
from collections import defaultdict
from pathlib import Path

'''
    v0.1.1 2025.10.25 @Gemini
    修正：使用 Path(__file__) 来定位项目根目录，确保绝对路径始终正确。
    (原始脚本逻辑依赖于 CWD，会导致路径错误)
'''

def convert_to_absolute_paths(data, project_root):
    """(已修正) 将 'images' 字段中的路径转换为绝对路径"""
    for item in data:
        absolute_images = []
        for relative_path in item["images"]:
            # relative_path 类似 "./datasets/VLAD_Remote/xView/images/train/1607_512_0_575.jpg"
            # project_root 是 /home/zirui/.../Qwen2.5-VL-FT-Remote
            clean_relative_path = relative_path.replace("./", "")
            abs_path = os.path.abspath(os.path.join(project_root, clean_relative_path))
            absolute_images.append(abs_path)
        item["images"] = absolute_images
    return data

def split_dataset(input_json_path, project_root, train_ratio=0.9):
    """Split dataset into train and test sets, grouped by images"""
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # 传入 project_root，而不是依赖 base_dir
    data = convert_to_absolute_paths(data, project_root)
    
    image_groups = defaultdict(list)
    for item in data:
        primary_image = item["images"][0]
        image_groups[primary_image].append(item)
    
    unique_images = list(image_groups.keys())
    random.shuffle(unique_images)
    
    split_idx = int(len(unique_images) * train_ratio)
    train_images = unique_images[:split_idx]
    test_images = unique_images[split_idx:]
    
    train_data = []
    test_data = []
    
    for img in train_images:
        train_data.extend(image_groups[img])
    
    for img in test_images:
        test_data.extend(image_groups[img])
    
    return train_data, test_data, test_images

def save_json(data, output_path):
    """Save data in compact JSON format with one sample per line"""
    # (保持你原来的 save_json 逻辑，输出标准 JSON 数组)
    with open(output_path, 'w') as f:
        f.write("[\n")
        for i, item in enumerate(data):
            json.dump(item, f, separators=(',', ':'))
            if i < len(data) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("]\n")

def save_test_image_list(test_images, output_dir):
    """Save test image paths to a text file"""
    output_path = Path(output_dir) / "test_image_list.txt"
    with open(output_path, 'w') as f:
        for img_path in test_images:
            f.write(f"{img_path}\n")

if __name__ == "__main__":
    # (已修正) 使用 Path(__file__) 确定项目根目录
    # 这使得无论你从哪里运行脚本，路径都是正确的
    
    # .../Qwen2.5-VL-FT-Remote/tools/dataset_Remote/process_vlad_remote.py
    SCRIPT_PATH = Path(__file__).resolve()
    # .../Qwen2.5-VL-FT-Remote/
    PROJECT_ROOT = SCRIPT_PATH.parent.parent.parent 
    
    input_json = PROJECT_ROOT / "datasets/VLAD_Remote/VLAD_Remote.json"
    output_dir = PROJECT_ROOT / "datasets/VLAD_Remote/"
    
    train_data, test_data, test_images = split_dataset(input_json, PROJECT_ROOT)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    train_json_path = Path(output_dir) / "VLAD_Remote_train.json"
    test_json_path = Path(output_dir) / "VLAD_Remote_test.json"
    
    save_json(train_data, train_json_path)
    save_json(test_data, test_json_path)
    save_test_image_list(test_images, output_dir)
    
    print(f"Split complete!\nTrain samples: {len(train_data)}\nTest samples: {len(test_data)}")
    print(f"Train data saved to: {train_json_path}")
    print(f"Test data saved to: {test_json_path}")
    print(f"Test image paths saved to: {Path(output_dir) / 'test_image_list.txt'}")