import os
import sys
import json
import time
from typing import List
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from tools.bbox_detector import BBoxDetector

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class BBoxEvaluator:
    def __init__(
        self,
        image_list_path: str,
        class_list: List[str],
        output_dir: str = "./results/eval/labels",
        task_config: str = "fixed",  # "fixed", "open", "open_ended"
        instructions_path: str = None,
        timeout_seconds: int = 60  # ⏰ 每张图最大推理时长
    ):
        self.image_list_path = image_list_path
        self.class_list = class_list
        self.task_config = task_config.lower()
        self.instructions_path = instructions_path
        self.timeout_s = timeout_seconds

        if self.task_config not in ["fixed", "open", "open_ended"]:
            raise ValueError("task_config must be either 'fixed', 'open', or 'open_ended'")

        # open 模式下默认类别
        if self.task_config == "open":
            self.class_list = [
                "facility",
                "structure",
                "passenger vehicle",
                "transportation",
                "cargo truck",
                "heavy vehicle"
            ]

        self.detector = BBoxDetector()
        
        # [!!! 关键修复 1 !!!]
        # 你的 06/07 脚本在创建文件夹时，会把绝对路径当作名字
        # 我们在这里也必须保持一致，否则 06/07 脚本找不到 05 脚本的输出
        model_name_as_path = self.detector.config['model_name'].lstrip('/')
        
        self.output_dir = f"{output_dir}/{model_name_as_path}/{self.task_config}"
        os.makedirs(self.output_dir, exist_ok=True)

        # open_ended 任务加载指令
        if self.task_config == "open_ended":
            if not instructions_path:
                raise ValueError("instructions_path must be provided for open_ended task")
            with open(instructions_path) as f:
                self.instructions = json.load(f)
            self.instruction_map = {item['image_path']: item for item in self.instructions}

    def generate_prompt(self, class_name: str = None, instruction_data: dict = None) -> str:
        """
        [!!! 关键修复 2 !!!]
        生成与 bbox_detector.py 解析器相匹配的简单提示词。
        我们不再使用复杂的 'bbox_2d' 格式，而是使用我们用 test_single_image.py 验证过的简单格式。
        """
        
        # 这是我们知道模型能回答、并且 bbox_detector.py 能解析的格式
        JSON_FORMAT_PROMPT = " and return bounding boxes in JSON format."

        if self.task_config in ["fixed", "open"]:
            # 使用我们 test_single_image.py 中验证过的简单提示词
            return f"This is aerial image, detect {class_name} in the image{JSON_FORMAT_PROMPT}"
        
        elif self.task_config == "open_ended":
            instruction = instruction_data.get('instruction', '')
            # 组合指令，并同样要求我们能解析的 JSON 格式
            # 例如: "Monitor the area ... for unusual activity, and return bounding boxes in JSON format."
            return f"{instruction}{JSON_FORMAT_PROMPT}"

    def _detect_with_timeout(self, image_path: str, prompt: str, timeout_s: float):
        """带超时控制的 detect 调用"""
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(self.detector.detect, image_path, prompt)
            try:
                # 确保超时至少为 1 秒
                return fut.result(timeout=max(1, int(timeout_s)))
            except FuturesTimeoutError:
                return {"__timeout__": True}

    def process_image(self, image_path: str) -> dict:
        """处理单张图片（超时则抛出 TimeoutError）"""
        start_t = time.time()

        def remaining_time():
            return self.timeout_s - (time.time() - start_t)

        results = {"image_path": image_path, "detections": {}, "task_config": self.task_config}

        if self.task_config in ["fixed", "open"]:
            for class_name in self.class_list:
                if remaining_time() <= 0:
                    raise TimeoutError(f"Timeout on image {image_path}")

                prompt = self.generate_prompt(class_name)
                det = self._detect_with_timeout(image_path, prompt, remaining_time())

                if det.get("__timeout__"):
                    raise TimeoutError(f"Timeout on detect for {image_path} (class={class_name})")

                # 我们只保存非空的 bboxes 列表
                if "bboxes" in det and det["bboxes"]:
                    results["detections"][class_name] = det["bboxes"]

        elif self.task_config == "open_ended":
            instruction_data = self.instruction_map.get(image_path)
            if not instruction_data:
                print(f"Warning: No instruction found for image {image_path}")
                return results

            results.update({
                "instruction": instruction_data.get('instruction'),
                "dataset_type": instruction_data.get('dataset_type'),
                "gt_object": instruction_data.get('objects')
            })

            if remaining_time() <= 0:
                raise TimeoutError(f"Timeout on image {image_path}")

            prompt = self.generate_prompt(instruction_data=instruction_data)
            det = self._detect_with_timeout(image_path, prompt, remaining_time())

            if det.get("__timeout__"):
                raise TimeoutError(f"Timeout on detect for {image_path} (open_ended)")

            # [!!! 关键修复 3 !!!]
            # open_ended 任务的 'detections' 是一个列表, 不是字典
            results["detections"] = det.get("bboxes", [])
            if "object" in det:
                results["object"] = det["object"]

        return results

    def save_results(self, image_path: str, results: dict):
        """保存检测结果为 JSON"""
        image_name = Path(image_path).stem
        output_path = os.path.join(self.output_dir, f"{image_name}.json")

        # 简单的 JSON 保存逻辑
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error saving JSON {output_path}: {e}")


    def run_evaluation(self):
        """遍历所有图片并评估"""
        with open(self.image_list_path) as f:
            image_paths = [line.strip() for line in f if line.strip()]

        for image_path in tqdm(image_paths, desc=f"Processing {self.task_config} images"):
            if not os.path.exists(image_path):
                print(f"Warning: Image not found - {image_path}")
                continue

            try:
                with self.detector:
                    results = self.process_image(image_path)
                    self.save_results(image_path, results)
            except TimeoutError as te:
                print(f"[TIMEOUT] {str(te)} -> discard {image_path}")
                continue
            except Exception as e:
                print(f"[ERROR] {image_path}: {str(e)}")
                continue


if __name__ == "__main__":
    IMAGE_LIST = "./datasets/VLAD_Remote/test_image_list.txt"
    INSTRUCTIONS_PATH = "./tools/dataset_Remote/instructions_with_objects.json"
    
    # [!!! 关键修复 4 !!!]
    # 你的 06/07 脚本是分开跑的，但 05 脚本是合在一起的。
    # 我们一次只跑一个任务，避免混淆。
    
    # 提示：你可以注释掉不想跑的任务
    
    print("\n--- [1/3] 正在运行 'fixed' 任务评测 ---")
    fixed_classes = ["Car", "Bus", "Truck", "Building"]
    fixed_eval = BBoxEvaluator(IMAGE_LIST, fixed_classes, task_config="fixed", timeout_seconds=60)
    fixed_eval.run_evaluation()

    print("\n--- [2/3] 正在运行 'open' 任务评测 ---")
    open_eval = BBoxEvaluator(IMAGE_LIST, [], task_config="open", timeout_seconds=60)
    open_eval.run_evaluation()

    print("\n--- [3/3] M正在运行 'open_ended' 任务评测 ---")
    open_ended_eval = BBoxEvaluator(
        IMAGE_LIST,
        [],
        task_config="open_ended",
        instructions_path=INSTRUCTIONS_PATH,
        timeout_seconds=60
    )
    open_ended_eval.run_evaluation()
    
    print("\n--- 所有评测推理已完成 ---")
    print("现在你可以运行 06 和 07 脚本来处理结果。")
