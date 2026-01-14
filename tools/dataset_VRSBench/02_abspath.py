import os
import json

def convert_to_absolute_paths(data, base_dir):
    """Convert image paths in JSON data to absolute paths"""
    for item in data:
        if "image" in item:  # Note: your JSON uses "image" not "images"
            relative_path = item["image"]
            # Join with base_dir directly (no need for os.path.dirname)
            abs_path = os.path.abspath(os.path.join(base_dir, relative_path.replace("./", "")))
            item["image"] = abs_path
    return data

def process_json_file(input_file, output_file, base_dir):
    """Process a JSON file and save with absolute paths"""
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Convert paths
    converted_data = convert_to_absolute_paths(data, base_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the converted data
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Converted data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Configuration
    input_json = "./datasets/VRSBench/VRSBench_train.json"          # Your input JSON file
    output_json = "./datasets/VRSBench/VRSBench.json"  # Output path in new folder
    image_base_dir = "./datasets/VRSBench/Images_train"  # Base directory where images are stored

    # Process the file
    process_json_file(input_json, output_json, image_base_dir)
    # 读取原始文件
    with open("./datasets/VRSBench/VRSBench.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取前500行
    top_500 = data[:]

    # 保存为新文件
    with open("./datasets/VRSBench/VRSBench_train_abs.json", 'w', encoding='utf-8') as f:
        json.dump(top_500, f, ensure_ascii=False, indent=4)

    print("前500行已成功保存为 VRSBench_train_top500.json")