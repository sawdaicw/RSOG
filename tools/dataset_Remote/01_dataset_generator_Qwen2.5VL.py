import os
import json
import yaml
from PIL import Image
from abc import ABC, abstractmethod
import random
import time
from tqdm import tqdm
import math
'''
    v0.1.0 2025.06.12 @jialei
    生成VLAD_Remote数据集的对话样本
    读取：各个数据集的图片和标注框
    生成：针对该数据集的Grounding任务对话样本，每种对话一个样本
    同时计算正样本和负样本数量，自动进行正：负 = 9：1的平衡
    生成的样本回答符合sharegpt格式，bbox包含三个属性：id、class和bbox_2d
    format of Qwen2.5-VL-XX
    
    v0.1.1 2025.06.16 @jialei
    生成数据格式由sharegpt格式调整为swift框架通用的message格式，用于grounding任务的微调
    生成的样本回答符合新的格式规范：
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "<image>找到图像中的<ref-object>"},
            {"role": "assistant", "content": "<bbox><bbox>"}
        ],
        "images": ["/xxx/x.jpg"],
        "objects": {
            "ref": ["羊"],
            "bbox": [[90.9, 160.8, 135, 212.8], [360.9, 480.8, 495, 532.8]]
        }
    }

'''
class BaseDatasetGenerator(ABC):
    """数据集生成器基类（带进度显示）"""
    
    def __init__(self, image_dir, label_dir, yaml_file, output_file):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.yaml_file = yaml_file
        self.output_file = output_file
        self.class_mapping = None
        self.data_entries = []
        self.none_count = 0
        self.total_conversations = 0
        self.processed_images = 0
        self.start_time = None
        self.last_update_time = None
        self.last_processed_count = 0
    
    def load_class_names(self):
        """加载YAML类别映射"""
        with open(self.yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        if 'names' in data and isinstance(data['names'], list):
            return {i: name for i, name in enumerate(data['names'])}
        elif 'names' in data and isinstance(data['names'], dict):
            return data['names']
        else:
            raise ValueError("Invalid YAML format: 'names' not found or invalid format")
    
    @abstractmethod
    def get_image_files(self):
        """获取图像文件列表（子类必须实现）"""
        pass
    
    @abstractmethod
    def parse_label_line(self, line):
        """解析标签行（子类必须实现）"""
        pass
    
    def print_progress(self, total_images):
        """实时打印处理进度"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        processed = self.processed_images
        
        # 计算处理速度（图片/秒）
        if self.last_update_time:
            time_diff = current_time - self.last_update_time
            processed_diff = processed - self.last_processed_count
            speed = processed_diff / time_diff if time_diff > 0 else 0
        else:
            speed = 0
            
        self.last_update_time = current_time
        self.last_processed_count = processed
        
        # 计算剩余时间
        remaining = (total_images - processed) / speed if speed > 0 else 0
        
        print(f"\rProcessing: {processed}/{total_images} images | "
              f"Speed: {speed:.1f} img/s | "
              f"Elapsed: {elapsed:.1f}s | "
              f"ETA: {remaining:.1f}s | "
              f"Valid: {self.total_conversations - self.none_count} | "
              f"None: {self.none_count}", end='', flush=True)
    
    def yolo_to_abs(self, coords, img_path):
        """将YOLO格式转换为绝对坐标（整数）"""
        try:
            with Image.open(img_path) as img:
                img.verify()
                img_width, img_height = img.size
        except (IOError, SyntaxError) as e:
            print(f"\nError opening image {img_path}: {str(e)}")
            return None
        
        if len(coords) == 4:  # HBB
            center_x, center_y, width, height = map(float, coords)
            x1 = int((center_x - width / 2) * img_width)
            y1 = int((center_y - height / 2) * img_height)
            x2 = int((center_x + width / 2) * img_width)
            y2 = int((center_y + height / 2) * img_height)
            return [x1, y1, x2, y2]
        elif len(coords) == 5:  # OBB
            center_x, center_y, width, height, angle = map(float, coords)
            x1 = int((center_x - width / 2) * img_width)
            y1 = int((center_y - height / 2) * img_height)
            x2 = int((center_x + width / 2) * img_width)
            y2 = int((center_y + height / 2) * img_height)
            return [x1, y1, x2, y2, int(angle)]
        else:
            return None


    def generate_conversation(self, class_name, label_list, img_path, task_augmentation=0, style_augmentation=0):
        """生成对话内容（新格式）"""
        # System message remains constant
        system_message = {"role": "system", "content": "You are a helpful assistant."}
        class_hierarchical_attributes = {
            # 1. 航空器类
            "Aerial Vehicles": {
                "descriptive_attributes": "aerodynamic structures with wings/rotors",
                "function_description": "airborne transportation/surveillance",
                "contextual_location": "airspace/airfields/helipads",
                "semantic_class": ["aircraft", "aviation systems", "flying vehicles", "aerial transport", "airborne machines", "flight systems"],
                "contains": [
                    "plane", "airplane", "Fixed-wing Aircraft", "Small Aircraft",
                    "Cargo Plane", "helicopter", "Helicopter", "Aircraft Hangar", "Helipad"
                ]
            },
            
            # 2. 车辆类
            "Ground Vehicles": {
                "descriptive_attributes": "wheeled motorized/non-motorized transport",
                "function_description": "road/rail transportation",
                "contextual_location": "roads/railways/parking areas/bike lanes",
                "semantic_class": ["motor vehicles", "road transport", "land vehicles", "automotive systems", "surface transport", "wheeled machines"],
                "contains": [
                    "large vehicle", "small vehicle", "small-vehicle", "large-vehicle",
                    "car", "van", "truck", "bus", "motor", "Passenger Vehicle", 
                    "Small Car", "Bus", "Pickup Truck", "Utility Truck", "Truck",
                    "Cargo Truck", "Truck w/Box", "Truck Tractor", "Trailer",
                    "Truck w/Flatbed", "Truck w/Liquid", "Crane Truck", 
                    "Railway Vehicle", "Passenger Car", "Cargo Car", "Flat Car",
                    "Tank car", "Locomotive", "tricycle", "awning-tricycle", "bicycle"
                ]
            },
            
            # 3. 船舶类
            "Maritime Vessels": {
                "descriptive_attributes": "watercraft with hull and superstructure",
                "function_description": "waterborne transportation/operations",
                "contextual_location": "waterways/harbors",
                "semantic_class": ["watercraft", "marine transport", "sea vessels", "nautical systems", "floating transport", "waterborne vehicles"],
                "contains": [
                    "ship", "Maritime Vessel", "Motorboat", "Sailboat", "Tugboat",
                    "Barge", "Fishing Vessel", "Ferry", "Yacht", "Container Ship",
                    "Oil Tanker"
                ]
            },
            
            # 4. 工程机械类
            "Engineering Equipment": {
                "descriptive_attributes": "heavy-duty movable machinery",
                "function_description": "construction/industrial operations",
                "contextual_location": "construction sites/industrial zones",
                "semantic_class": ["construction machinery", "heavy equipment", "engineering vehicles", "industrial machines", "earth-moving systems", "site equipment"],
                "contains": [
                    "container crane", "Container Crane", "Engineering Vehicle",
                    "Tower crane", "Reach Stacker", "Straddle Carrier", "Mobile Crane",
                    "Dump Truck", "Haul Truck", "Scraper/Tractor", 
                    "Front loader/Bulldozer", "Excavator", "Cement Mixer", 
                    "Ground Grader"
                ]
            },
            
            # 5. 基础设施类
            "Infrastructure": {
                "descriptive_attributes": "static man-made structures",
                "function_description": "supporting transportation/utility systems",
                "contextual_location": "urban/rural landscapes",
                "semantic_class": ["civil structures", "public works", "built environment", "urban installations", "static constructions", "utility systems"],
                "contains": [
                    "storage tank", "storage-tank", "Storage Tank", "container",
                    "Shipping Container", "Shipping container lot", "bridge",
                    "roundabout", "harbor", "Pylon", "Tower", "windmill"
                ]
            },
            
            # 6. 运动场地类
            "Sports Facilities": {
                "descriptive_attributes": "standardized playing surfaces with markings",
                "function_description": "athletic/recreational activities",
                "contextual_location": "recreational complexes",
                "semantic_class": ["athletic venues", "recreational facilities", "sports grounds", "playing surfaces", "competition areas", "game spaces"],
                "contains": [
                    "baseball diamond", "tennis court", "basketball court",
                    "ground track field", "soccer ball field", "swimming pool",
                    "swimming-pool"
                ]
            },
            
            # 7. 建筑结构类
            "Buildings": {
                "descriptive_attributes": "enclosed architectural structures",
                "function_description": "human habitation/activities",
                "contextual_location": "residential/commercial areas",
                "semantic_class": ["architectural structures", "constructed spaces", "human habitats", "roofed buildings", "permanent shelters", "enclosed edifices"],
                "contains": [
                    "Hut/Tent", "Shed", "Building", "Aircraft Hangar",
                    "Damaged Building", "Facility", "Construction Site", "Vehicle Lot"
                ]
            },
            
            # 8. 人员类
            "People": {
                "descriptive_attributes": "human figures with distinguishable body posture",
                "function_description": "pedestrian activities",
                "contextual_location": "sidewalks/urban areas",
                "semantic_class": ["human beings", "individual persons", "pedestrian entities", "social figures", "mobile agents", "upright organisms"],
                "contains": [
                    "pedestrian", "people"
                ]
            }
        }
        # Define command templates
        commands = {
            # English commands
            "formal": "Process the provided aerial image, identify all instances of {class_name}, and return their bounding box coordinates in JSON format.",
            "concise": "Detect {class_name} in this aerial image. Output bounding boxes as JSON.",
            "technical": "Run object detection on this aerial image for class {class_name}. Format results as JSON with bounding box.",
            "task_oriented": "Analyze the aerial image and generate a JSON file containing bounding boxes for all detected {class_name} objects.",
            "friendly": "Hey, could you scan this aerial image for {class_name} and give me the bounding boxes in JSON? Thanks!",
            "military": "Mission: Detect and mark all {class_name} in the aerial recon image. Report coordinates in JSON format.",
            "debug": "DEBUG: Perform inference on aerial image for class {class_name}. Dump bbox predictions in JSON.",
            "script_like": "detect --image aerial --class {class_name} --format json",
            "academic": "Please apply the detection model to the aerial imagery and output the bounding box annotations of {class_name} in JSON for further analysis.",
            "urgent": "Priority request: Identify all {class_name} in this aerial feed ASAP! JSON bbox output required.",

            # # Chinese translations
            # "formal_zh": "请处理提供的航拍图像，识别所有 {class_name} 目标，并以JSON格式返回其边界框坐标。",
            # "concise_zh": "检测航拍图中的 {class_name}，输出边界框JSON。",
            # "technical_zh": "对航拍图像执行目标检测，类别为 {class_name}。结果以JSON格式输出边界框坐标。",
            # "task_oriented_zh": "分析航拍图像并生成JSON文件，包含所有检测到的 {class_name} 目标的边界框。",
            # "friendly_zh": "帮忙扫描这张航拍图里的 {class_name}，用JSON格式返回边界框，谢谢！",
            # "military_zh": "任务指令：检测并标记航拍侦察图像中所有 {class_name} 目标，以JSON格式上报坐标。",
            # "debug_zh": "调试：对航拍图中的 {class_name} 执行推理，将边界框预测结果导出为JSON。",
            # "script_like_zh": "检测 --图像 航拍 --类别 {class_name} --格式 json",
            # "academic_zh": "请对航拍影像应用检测模型，输出 {class_name} 的边界框标注（JSON格式）以供后续分析。",
            # "urgent_zh": "紧急请求：快速识别航拍画面中的所有 {class_name}！需JSON格式边界框输出。"
        }
        
        task_augmentation_commands = {
            # Implicit commands (indirect but clear targeting)
            "implicit_descriptive": "From this aerial imagery, detect all objects matching: {descriptive_attributes}. Output bounding boxes in JSON.",  # (e.g. "4-wheeled motorized vehicles" for cars)
            "implicit_functional": "In this drone footage, identify objects capable of {function_description}. Provide JSON bboxes.",  # (e.g. "road transportation" for cars)
            "implicit_environmental": "In aerial view, mark objects typically found in {contextual_location} zones. Export as JSON.",  # (e.g. "parking" for cars)
            "implicit_semantic": "Analyze this overhead view and locate objects related to {semantic_class}. Return JSON annotations.",  # (e.g. "sedan" for cars)
            
            # Ambiguous commands (multiple interpretations possible)
            "ambiguous_priority": "From this aerial survey, detect high-value targets. Format results as JSON.",  # (class-dependent interpretation)
            "ambiguous_status": "Locate objects requiring immediate attention in the drone imagery. Output JSON bboxes.",  # (could mean vehicles, animals, etc.)
            "ambiguous_operational": "From overhead view, find mission-relevant objects. Save detections in JSON.",  # (requires context)
            
            # Confusing commands (unclear intent)
            "confusing_vague": "Review this aerial grid and report findings. JSON format.",  # (no detection specifics)
            "confusing_metaphorical": "In this drone capture, find things that catch your eye. Output as JSON.",  # (subjective)
            "confusing_openended": "Analyze this aerial perspective. Show me what's there in JSON bboxes.",  # (no guidance)
            
            # # Chinese translations
            # "implicit_descriptive_zh": "基于航拍图像，检测所有符合特征的目标：{descriptive_attributes}，输出JSON边界框。",
            # "implicit_functional_zh": "在无人机画面中，识别能够{function_description}的目标，提供JSON边界框。",
            # "implicit_environmental_zh": "在俯视图中，标记{contextual_location}区域的典型目标，导出为JSON。",
            # "implicit_semantic_zh": "分析俯视图并定位与{class_name}语义相关的目标，返回JSON标注。",
            # "ambiguous_priority_zh": "通过航拍扫描，检测高价值目标，结果格式化为JSON。",
            # "ambiguous_dynamic_zh": "在航拍图像中，识别感兴趣的移动目标，输出JSON边界框。",
            # "ambiguous_operational_zh": "从俯视视角查找任务相关目标，检测结果保存为JSON。",
            # "confusing_vague_zh": "查看航拍网格并报告发现，JSON格式。",
            # "confusing_metaphorical_zh": "在无人机拍摄画面中，找出引人注目的目标，输出为JSON。",
            # "confusing_openended_zh": "分析航拍视角，用JSON边界框显示内容。"
        }
        
        # Determine which augmentation to apply
        style_content = ""
        rand_val = random.random()
        
        if rand_val < task_augmentation:
            # Apply task augmentation
            selected_key = random.choice(list(task_augmentation_commands.keys()))
            template = task_augmentation_commands[selected_key]
            
            # Get the hierarchical attributes for the class (if needed)
            class_info = None
            if any(ph in template for ph in ["{descriptive_attributes}", "{function_description}", 
                                            "{contextual_location}", "{semantic_class}"]):
                for category in class_hierarchical_attributes.values():
                    if class_name in category["contains"]:
                        class_info = category
                        break
            
            # Handle different template types
            if "{descriptive_attributes}" in template and class_info:
                style_content = template.format(
                    descriptive_attributes=class_info["descriptive_attributes"]
                )
            elif "{function_description}" in template and class_info:
                style_content = template.format(
                    function_description=class_info["function_description"]
                )
            elif "{contextual_location}" in template and class_info:
                style_content = template.format(
                    contextual_location=class_info["contextual_location"]
                )
            elif "{semantic_class}" in template and class_info:
                # Select a random semantic class from the list
                semantic_class = random.choice(class_info["semantic_class"])
                style_content = template.format(
                    semantic_class=semantic_class
                )
            elif "{class_name}" in template:
                style_content = template.format(class_name=class_name)
            else:
                # For commands without any placeholders (ambiguous/confusing commands)
                style_content = template
        elif rand_val < task_augmentation + style_augmentation:
            # Apply style augmentation
            selected_key = random.choice(list(commands.keys()))
            style_content = commands[selected_key].format(class_name=class_name)
        else:
            # Original format
            style_content = f"This is aerial image, detect {class_name} in the image and return bounding boxes in JSON format."
        
        # User message with image and reference object
        user_message = {"role": "user", "content": style_content}
        
        # Assistant message with bounding boxes
        if label_list:
            # Format bounding boxes as JSON objects with ref-object and id
            bbox_placeholders = ",".join(["<bbox>" for _ in label_list])
            assistant_content = '{"object": "<ref-object>", "bbox": [' + bbox_placeholders + ']}'
        else:
            assistant_content = "None"
        
        assistant_message = {"role": "assistant", "content": assistant_content}
        
        # Prepare objects dictionary
        objects_dict = {
            "ref": [class_name],
            "bbox": [obj["bbox_2d"] for obj in label_list] if label_list else []
        }
        
        return {
            "messages": [system_message, user_message, assistant_message],
            "images": [img_path],
            "objects": objects_dict
        }, assistant_content == "None"
    
    def process_image(self, img_file, total_images):
        """处理单张图像"""
        self.processed_images += 1
        if self.processed_images % 10 == 0 or self.processed_images == total_images:
            self.print_progress(total_images)
        
        img_id = os.path.splitext(img_file)[0]
        img_path = os.path.join(self.image_dir, img_file)
        
        # 验证图像并获取原始尺寸
        try:
            with Image.open(img_path) as img:
                img.verify()
                orig_width, orig_height = img.size
        except (IOError, SyntaxError) as e:
            print(f"\nInvalid image file {img_file}, skipping...")
            return
        
        # 读取标签文件
        label_file = os.path.join(self.label_dir, f"{img_id}.txt")
        label_lines = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                label_lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # 收集对象
        class_objects = {class_name: [] for class_name in self.class_mapping.values()}
        
        for line in label_lines:
            try:
                class_name, coords = self.parse_label_line(line)
                abs_coords = self.yolo_to_abs(coords, img_path)
                if abs_coords is not None:
                    # 直接使用原始坐标，不进行任何转换
                    if len(abs_coords) == 4:  # HBB
                        obj_dict = {
                            "id": len(class_objects[class_name]) + 1,
                            "class": class_name,
                            "bbox_2d": abs_coords  # 已经是整数，直接使用
                        }
                    else:  # OBB
                        obj_dict = {
                            "id": len(class_objects[class_name]) + 1,
                            "class": class_name,
                            "bbox_2d": abs_coords[:4] + [abs_coords[4]]  # 前4个坐标已经是整数，角度可能是整数或浮点数
                        }
                    class_objects[class_name].append(obj_dict)
            except ValueError as e:
                print(f"\nSkipping invalid line: {line} - {str(e)}")
                continue
        
        # 生成样本条目
        for class_name in self.class_mapping.values():
            objects = class_objects[class_name]
            
            data_entry, is_none = self.generate_conversation(class_name, objects, img_path)
            self.total_conversations += 1
            if is_none:
                self.none_count += 1
            
            self.data_entries.append({
                "id": f"{img_id}_{class_name}",
                **data_entry
            })
    
    def generate(self):
        """生成数据集"""
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        self.class_mapping = self.load_class_names()
        image_files = self.get_image_files()
        total_images = len(image_files)
        
        print(f"\nStarting processing {total_images} images...")
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        # 使用tqdm显示进度条
        for img_file in tqdm(image_files, desc="Processing images"):
            self.process_image(img_file, total_images)
        
        # 写入JSON文件
        with open(self.output_file, 'w') as f:
            for entry in self.data_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        
        # 打印统计信息
        print(f"\n\nDataset generation complete: {self.output_file}")
        print(f"Total images processed: {len(set(e['id'].split('_')[0] for e in self.data_entries))}")
        print(f"Total conversations: {self.total_conversations}")
        print(f"None responses: {self.none_count} ({(self.none_count/self.total_conversations)*100:.2f}%)")
        print(f"Valid responses: {self.total_conversations - self.none_count} ({(1 - self.none_count/self.total_conversations)*100:.2f}%)")


class VisDroneGenerator(BaseDatasetGenerator):
    """VisDrone数据集生成器"""
    
    def get_image_files(self):
        return sorted([f for f in os.listdir(self.image_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png', '.tif'))])
    
    def parse_label_line(self, line):
        parts = line.split()
        if len(parts) not in [5, 6]:
            raise ValueError("Invalid label format")
        
        class_id = int(parts[0])
        if class_id not in self.class_mapping:
            raise ValueError(f"Class ID {class_id} not found in mapping")
        
        return self.class_mapping[class_id], parts[1:5]  # 只取前4个坐标(center_x, center_y, width, height)


class XViewGenerator(BaseDatasetGenerator):
    """xView数据集生成器（支持父类包含子类）"""
    
    def __init__(self, image_dir, label_dir, yaml_file, output_file, mode=2):
        """
        参数:
            mode: 1 - 同时检测父类和子类 (默认)
                 2 - 只检测父类 (子类合并到父类)
        """
        super().__init__(image_dir, label_dir, yaml_file, output_file)
        self.mode = mode

        # 父类-子类映射
        self.parent_child_mapping = {
            "Fixed-wing Aircraft": ["Small Aircraft", "Cargo Plane", "Passenger/Cargo Plane"],
            "Passenger Vehicle": ["Small Car", "Bus", "Passenger Car"],
            "Truck": ["Pickup Truck", "Utility Truck", "Cargo Truck", "Truck w/Box", 
                     "Truck Tractor", "Trailer", "Truck w/Flatbed", "T Truck w/Liquid"],
            "Railway Vehicle": ["Cargo Car", "Flat Car", "Tank car", "Locomotive"],
            "Engineering Vehicle": ["Crane Truck", "Dump Truck", "Haul Truck", "Scraper/Tractor",
                                  "Front loader/Bulldozer", "Excavator", "Cement Mixer", "Ground Grader"],
            "Maritime Vessel": ["Motorboat", "Sailboat", "Tugboat", "Barge", "Fishing Vessel",
                               "Ferry", "Yacht", "Container Ship", "Oil Tanker"],
            "Building": ["Aircraft Hangar", "Damaged Building", "Facility", "Hut/Tent", "Shed"]
        }
        
        # 模式二专用：子类到父类的反向映射
        self.child_to_parent = {}
        for parent, children in self.parent_child_mapping.items():
            for child in children:
                self.child_to_parent[child] = parent

    def get_image_files(self):
        return sorted([f for f in os.listdir(self.image_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png', '.tif'))])
    
    def parse_label_line(self, line):
        parts = line.split()
        if len(parts) not in [5, 6]:  # xView使用固定6参数格式 (class_id, x, y, w, h, angle)
            raise ValueError("xView requires 6 values per line")
        
        class_id = int(parts[0])
        if class_id not in self.class_mapping:
            raise ValueError(f"Class ID {class_id} not found in mapping")
        
        return self.class_mapping[class_id], parts[1:6]  # 取5个坐标(center_x, center_y, width, height, angle)
    
    def process_image(self, img_file, total_images):
        """重写处理逻辑以支持两种模式"""
        self.processed_images += 1
        if self.processed_images % 10 == 0 or self.processed_images == total_images:
            self.print_progress(total_images)
        
        img_id = os.path.splitext(img_file)[0]
        img_path = os.path.join(self.image_dir, img_file)
        
        # 验证图像
        try:
            with Image.open(img_path) as img:
                img.verify()
        except (IOError, SyntaxError) as e:
            print(f"Invalid image file {img_file}, skipping...")
            return
        
        # 读取标签文件
        label_file = os.path.join(self.label_dir, f"{img_id}.txt")
        label_lines = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                label_lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # 收集所有对象（按实际类别）
        all_objects = {}
        for line in label_lines:
            try:
                class_name, coords = self.parse_label_line(line)
                
                # 模式二处理：如果是子类，转换为父类
                if self.mode == 2 and class_name in self.child_to_parent:
                    class_name = self.child_to_parent[class_name]
                
                abs_coords = self.yolo_to_abs(coords, img_path)
                if abs_coords is not None:
                    if class_name not in all_objects:
                        all_objects[class_name] = []
                    
                    if len(abs_coords) == 5:  # OBB
                        obj_dict = {
                            "id": len(all_objects[class_name]) + 1,
                            "class": class_name,
                            "bbox_2d": abs_coords[:4] + [abs_coords[4]]
                        }
                    else:  # HBB
                        obj_dict = {
                            "id": len(all_objects[class_name]) + 1,
                            "class": class_name,
                            "bbox_2d": abs_coords
                        }
                    all_objects[class_name].append(obj_dict)
            except ValueError as e:
                print(f"Skipping invalid line: {line} - {str(e)}")
                continue
        
        # 生成样本条目
        if self.mode == 1:
            # 模式一：保持现有逻辑，同时处理父类和子类
            for class_name in self.class_mapping.values():
                # 如果是父类，收集所有子类对象
                if class_name in self.parent_child_mapping:
                    child_classes = self.parent_child_mapping[class_name]
                    objects = []
                    for child in child_classes:
                        if child in all_objects:
                            objects.extend(all_objects[child])
                # 如果是独立类别或子类，直接使用
                else:
                    objects = all_objects.get(class_name, [])
                
                # 生成对话
                data_entry, is_none = self.generate_conversation(class_name, objects, img_path)
                self.total_conversations += 1
                if is_none:
                    self.none_count += 1
                
                self.data_entries.append({
                    "id": f"{img_id}_{class_name}",
                    **data_entry
                })
        else:
            # 模式二：只处理父类（子类已在上游合并）
            for class_name in self.parent_child_mapping.keys():
                objects = all_objects.get(class_name, [])
                
                # 生成对话
                data_entry, is_none = self.generate_conversation(class_name, objects, img_path)
                self.total_conversations += 1
                if is_none:
                    self.none_count += 1
                
                self.data_entries.append({
                    "id": f"{img_id}_{class_name}",
                    **data_entry
                })

def balance_and_merge_json(json_files, output_file="./datasets/VLAD_Remote/VLAD_Remote.json"):
    """
    处理JSON文件列表：
    1. 对每个文件平衡正负样本（9:1）
    2. 检查每个样本长度不超过2048
    3. 合并所有文件并保存为JSON数组格式，每个样本占一行
    """
    merged_data = []
    skipped_count = 0  # 记录因长度过长而被跳过的样本数

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 尝试读取整个文件作为JSON数组
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples = data
                    else:
                        samples = [data]
                except json.JSONDecodeError:
                    # 如果失败，尝试逐行读取
                    f.seek(0)
                    samples = []
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                samples.append(json.loads(line))
                            except json.JSONDecodeError:
                                print(f"跳过无效行: {line[:50]}...")
                                continue

            # 统计正负样本
            positive_samples = []
            negative_samples = []
            for sample in samples:
                try:
                    # 检查样本长度
                    sample_str = json.dumps(sample)
                    if len(sample_str) > 2048:
                        skipped_count += 1
                        continue
                    
                    assistant_content = sample["messages"][2]["content"]
                    if assistant_content.lower() != "none":
                        positive_samples.append(sample)
                    else:
                        negative_samples.append(sample)
                except (KeyError, IndexError) as e:
                    print(f"无效样本格式: {e}")
                    continue

            # 计算需保留的负样本数（正样本数的1/9）
            target_neg_count = len(positive_samples) // 9
            if target_neg_count > len(negative_samples):
                target_neg_count = len(negative_samples)

            # 随机选择负样本
            selected_negatives = random.sample(negative_samples, target_neg_count)

            # 合并正样本和选中的负样本
            merged_data.extend(positive_samples + selected_negatives)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            continue

    # 保存为JSON数组格式，每个样本占一行
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n')  # 开始数组
        # 写入每个样本，后面加逗号（除了最后一个）
        for i, entry in enumerate(merged_data):
            line = json.dumps(entry, separators=(',', ':'), ensure_ascii=False)
            if i < len(merged_data) - 1:
                line += ','
            f.write(line + '\n')
        f.write(']\n')  # 结束数组

    print(f"处理完成！共合并{len(merged_data)}条样本，保存至 {output_file}")
    print(f"因长度超过2048被跳过的样本数: {skipped_count}")
if __name__ == "__main__":
    # 处理VisDrone数据集
    for name in ['train', 'val']:
        generator = VisDroneGenerator(
            image_dir=f"./datasets/VLAD_Remote/VisDrone/VisDrone2019-DET-{name}/images",
            label_dir=f"./datasets/VLAD_Remote/VisDrone/VisDrone2019-DET-{name}/labels",
            yaml_file="./datasets/VLAD_Remote/VisDrone/VisDrone.yaml",
            output_file=f"./datasets/VLAD_Remote/visdrone_dataset_{name}.json"
        )
        generator.generate()
    
    # 处理xView数据集
    generator = XViewGenerator(
        image_dir="./datasets/VLAD_Remote/xView/images/train",
        label_dir="./datasets/VLAD_Remote/xView/labels/train",
        yaml_file="./datasets/VLAD_Remote/xView/xView.yaml",
        output_file="./datasets/VLAD_Remote/xView.json"
    )
    generator.generate()

    # # 处理DOTAv1.5数据集
    # for name in ['train', 'val']:
    #     generator = VisDroneGenerator(
    #         image_dir=f"./datasets/VLAD_Remote/DOTA15/images/{name}",
    #         label_dir=f"./datasets/VLAD_Remote/DOTA15/labels/{name}_rectangle",
    #         yaml_file="./datasets/VLAD_Remote/DOTA15/DOTAv1.5.yaml",
    #         output_file=f"./datasets/VLAD_Remote/DOTA15_dataset_{name}.json"
    #     )
    #     generator.generate()
    
    # # 处理soda数据集
    # for name in ['train', 'val', 'test']:
    #     generator = VisDroneGenerator(
    #         image_dir=f"./datasets/VLAD_Remote/soda-a-yolo8/{name}/images",
    #         label_dir=f"./datasets/VLAD_Remote/soda-a-yolo8/{name}/labels",
    #         yaml_file="./datasets/VLAD_Remote/soda-a-yolo8/soda-a.yaml",
    #         output_file=f"./datasets/VLAD_Remote/soda_dataset_{name}.json"
    #     )
    #     generator.generate()

    json_files = [
        # "./datasets/VLAD_Remote/DOTA15_dataset_train.json",
        # "./datasets/VLAD_Remote/DOTA15_dataset_val.json",
        "./datasets/VLAD_Remote/visdrone_dataset_train.json",
        "./datasets/VLAD_Remote/visdrone_dataset_val.json",
        "./datasets/VLAD_Remote/xView.json",
        # "./datasets/VLAD_Remote/soda_dataset_train.json",
        # "./datasets/VLAD_Remote/soda_dataset_test.json",
        # "./datasets/VLAD_Remote/soda_dataset_val.json"
        
        # 添加更多文件...
    ]
    balance_and_merge_json(json_files)