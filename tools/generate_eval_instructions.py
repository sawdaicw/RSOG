import json
import os
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
from tqdm import tqdm

# 数据集类别定义
DATASET_CATEGORIES = {
    'VisDrone': [
        'pedestrian', 'people', 'bicycle', 'car', 'van', 
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ],
    'xView': [
        'Fixed-wing Aircraft', 'Passenger Vehicle', 'Building','Truck','Railway Vehicle','Maritime Vessel','Engineering Vessel'
    ]
}

def encode_image(image):
    """将图片编码为base64字符串"""
    with BytesIO() as buffer:
        image.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def load_image_list(image_list_path):
    """加载图片列表"""
    with open(image_list_path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def get_dataset_type(image_path):
    """根据图像路径判断数据集类型"""
    if 'VisDrone' in image_path:
        return 'VisDrone'
    elif 'xView' in image_path:
        return 'xView'
    else:
        return None

def build_prompt(dataset_type):
    """构建对应数据集类型的提示词"""
    if dataset_type not in DATASET_CATEGORIES:
        return None
    
    categories = DATASET_CATEGORIES[dataset_type]
    categories_str = "`, `".join(categories)
    
    return f"""
    Give some instructions delivering specific purpose that require the use of the following objects.  
    To be effective, instructions should use a language that is natural and familiar, much like the way people give directions to robots.  
    For each instruction, return the related objects.  
    Constraints:  
    - The purpose and sentence structure of the instructions should be diverse to accommodate different scenarios.  
    - Instructions must align with UAV reconnaissance objectives (area surveillance, target tracking, etc.)
    - Select objects that are visually confirmed in the image
    - Allowed objects (exactly one object required per instruction): `{categories_str}`.
    - Format the results as following: Instruction: XXXX, Objects: ['object1', 'object2', ...].   
    Now, analyze this image and give me one instruction with related objects.
    """

def generate_instruction(image_path, client):
    """使用Qwen2.5vl-7b模型生成指令和GT目标"""
    try:
        image = Image.open(image_path)
        encoded_image = encode_image(image)
        
        dataset_type = get_dataset_type(image_path)
        if not dataset_type:
            print(f"Unknown dataset type for image: {image_path}")
            return None
        
        prompt = build_prompt(dataset_type)
        if not prompt:
            return None
        
        response = client.chat.completions.create(
            model= "Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]}],
            temperature=0.3,
            top_p=0.9,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def parse_response(response):
    """解析模型响应，提取指令和对象"""
    if not response:
        return None, None
    
    # 尝试标准格式解析
    try:
        if "Instruction:" in response and "Objects:" in response:
            # 分割指令部分
            instruction_part = response.split("Instruction:")[1]
            instruction = instruction_part.split("Objects:")[0].strip()
            
            # 分割对象部分
            objects_part = response.split("Objects:")[1].strip()
            
            # 安全地提取列表对象
            if objects_part.startswith('[') and objects_part.endswith(']'):
                objects = json.loads(objects_part)
            else:
                # 尝试处理不带方括号的情况
                objects = [obj.strip(" '\"") for obj in objects_part.split(",")]
            
            return instruction, objects
        elif "Instruction:" in response and "Object:" in response:
            # 分割指令部分
            instruction_part = response.split("Instruction:")[1]
            instruction = instruction_part.split("Object:")[0].strip()
            
            # 分割对象部分
            objects_part = response.split("Object:")[1].strip()
            
            # 安全地提取列表对象
            if objects_part.startswith('[') and objects_part.endswith(']'):
                objects = json.loads(objects_part)
            else:
                # 尝试处理不带方括号的情况
                objects = [obj.strip(" '\"") for obj in objects_part.split(",")]
            
            return instruction, objects
    except Exception as e:
        print(f"标准格式解析失败: {str(e)}")
    
    # 如果标准格式解析失败，尝试其他启发式方法
    try:
        # 尝试提取引号内的指令
        if '"' in response:
            parts = response.split('"')
            if len(parts) >= 3:
                instruction = parts[1]
                # 尝试在剩余文本中查找对象
                remaining_text = parts[2]
                objects = []
                for word in remaining_text.split():
                    clean_word = word.strip(".,'\"")
                    if clean_word in DATASET_CATEGORIES['VisDrone'] + DATASET_CATEGORIES['xView']:
                        objects.append(clean_word)
                if objects:
                    return instruction, objects
        
        # 最后尝试提取第一行作为指令
        lines = response.split('\n')
        instruction = lines[0].strip()
        objects = []
        for line in lines[1:]:
            for word in line.split():
                clean_word = word.strip(".,'\"")
                if clean_word in DATASET_CATEGORIES['VisDrone'] + DATASET_CATEGORIES['xView']:
                    objects.append(clean_word)
        if objects:
            return instruction, objects
        
        # 如果还是找不到对象，返回默认值
        return instruction, ["unknown"]
    
    except Exception as e:
        print(f"解析响应失败: {str(e)}")
        return None, None

def process_images(image_list, output_json_path, client):
    """处理所有图片并保存结果"""
    results = []
    
    for image_path in tqdm(image_list, desc="Processing images"):
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
            
        response = generate_instruction(image_path, client)
       
        instruction, objects = parse_response(response)
        print(response)
        if instruction and objects:
            dataset_type = get_dataset_type(image_path)
            result = {
                "image_path": image_path,
                "dataset_type": dataset_type,
                "instruction": instruction,
                "objects": objects
            }
            results.append(result)
    
    # 保存结果到JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_json_path}")
    return results

def main():
    # 配置OpenAI客户端（假设使用本地部署的vLLM服务）
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    # 图片列表路径和输出JSON路径
    image_list_path = "./datasets/VLAD_Remote/test_image_list.txt"
    output_json_path = "./results/instructions_with_objects.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    # 加载图片列表
    image_list = load_image_list(image_list_path)
    
    # 处理所有图片
    process_images(image_list, output_json_path, client)

if __name__ == "__main__":
    main()