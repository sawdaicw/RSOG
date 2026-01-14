import os
import sys
import json
from typing import List
from tqdm import tqdm
from pathlib import Path
import argparse
import re
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Import functions from the original inference code
from inference import CoordinatesQuantizer, extract_polygons, extract_roi

class FalconBBoxEvaluator:
    def __init__(
        self, 
        checkpoint_path: str,
        image_list_path: str, 
        class_list: List[str], 
        output_dir: str = "./results/eval/labels",
        task_config: str = "fixed",  # "fixed" or "open" or "open_ended"
        instructions_path: str = None  # Path to instructions JSON file
    ):
        """
        Initialize the Falcon evaluator
        
        :param checkpoint_path: Path to Falcon model checkpoint
        :param image_list_path: Path to text file containing image paths (one per line)
        :param class_list: List of classes to detect (used when task_config is "fixed")
        :param output_dir: Directory to save JSON results (default: ./results/eval/labels)
        :param task_config: "fixed" for fixed classes or "open" for open classes
        :param instructions_path: Path to JSON file containing instructions (for open_ended task)
        """
        self.checkpoint_path = checkpoint_path
        self.image_list_path = image_list_path
        self.class_list = class_list
        self.task_config = task_config.lower()
        self.instructions_path = instructions_path
        
        if self.task_config not in ["fixed", "open", "open_ended"]:
            raise ValueError("task_config must be either 'fixed', 'open', or 'open_ended'")
            
        # Define open classes if needed
        if self.task_config == "open":
            self.class_list = [
                "facility",
                "structure",
                "passenger vehicle",
                "transportation",
                "cargo truck",
                "heavy vehicle"
            ]
        
        # Initialize model and processor
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint_path,
            trust_remote_code=True,
        )
        
        # Coordinate quantizer
        self.coordinates_quantizer = CoordinatesQuantizer("floor", (1000, 1000))
        
        # Output directory setup
        self.output_dir = f"{output_dir}/Falcon/{self.task_config}"
        
        # Load instructions if open_ended task
        if self.task_config == "open_ended":
            if not instructions_path:
                raise ValueError("instructions_path must be provided for open_ended task")
            with open(instructions_path) as f:
                self.instructions = json.load(f)
            # Create mapping from image path to instruction for faster lookup
            self.instruction_map = {item['image_path']: item for item in self.instructions}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_prompt(self, class_name: str = None, instruction_data: dict = None) -> str:
        """Generate the detection prompt for a given class or for open detection"""
        if self.task_config == "fixed":
            return (
                f"""Detect all {class_name} in the image.\nUse horizontal bounding boxes."""
            )
        elif self.task_config == "open":
            return (
                f"""Detect all {class_name} in the image.\nUse horizontal bounding boxes."""
            )
        elif self.task_config == "open_ended":
            if not instruction_data:
                raise ValueError("instruction_data must be provided for open_ended task")
            
            instruction = instruction_data.get('instruction', '')
            objects = instruction_data.get('objects', [])
            dataset_type = instruction_data.get('dataset_type', '')
            
            return f"""
            Mission Instruction: {instruction}            
            Analyze this aerial image and detect all objects relevant to the mission instruction.
            Return the bounding boxes in format <x1><y1><x2><y2> with labels.
            Example: <100><200><150><250> for a car
            """
    
    def convert_relative_to_absolute(self, bbox, image_size):
        """
        Convert relative coordinates (0-1000) to absolute pixel coordinates
        
        :param bbox: List of relative coordinates [x1, y1, x2, y2]
        :param image_size: Tuple of (width, height) of the original image
        :return: List of absolute coordinates [x1, y1, x2, y2]
        """
        width, height = image_size
        x1, y1, x2, y2 = bbox
        
        # Convert relative coordinates to absolute
        x1_abs = int(x1 * width / 1000)
        y1_abs = int(y1 * height / 1000)
        x2_abs = int(x2 * width / 1000)
        y2_abs = int(y2 * height / 1000)
        
        return [x1_abs, y1_abs, x2_abs, y2_abs]
    
    def process_image(self, image_path: str) -> dict:
        """
        Process a single image and detect all classes
        
        :param image_path: Path to the image file
        :return: Dictionary containing all detection results for the image
        """
        results = {
            "image_path": image_path, 
            "detections": {}, 
            "task_config": self.task_config
        }
        
        image = Image.open(image_path).convert("RGB")
        image_size = (image.width, image.height)
        results["image_size"] = image_size  # Save image size for reference
        
        if self.task_config == "fixed":
            for class_name in self.class_list:
                prompt = self.generate_prompt(class_name)
                
                # Run Falcon inference
                inputs = self.processor(text=prompt, images=image, return_tensors="pt")
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=8192,
                    num_beams=3,
                    do_sample=False,
                )
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                print(generated_text)
                
                # Parse bounding boxes
                pred_bboxes = extract_roi(
                    generated_text, pattern=r"<(\d+)><(\d+)><(\d+)><(\d+)>"
                )
                
                # Convert to list of absolute bbox coordinates
                bboxes = []
                for bbox in pred_bboxes:
                    rel_bbox = [int(coord) for coord in bbox]
                    abs_bbox = self.convert_relative_to_absolute(rel_bbox, image_size)
                    bboxes.append(abs_bbox)
                
                results["detections"][class_name] = bboxes
                
        elif self.task_config == "open":
            for class_name in self.class_list:
                prompt = self.generate_prompt(class_name)
                
                # Run Falcon inference
                inputs = self.processor(text=prompt, images=image, return_tensors="pt")
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=8192,
                    num_beams=3,
                    do_sample=False,
                )
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                
                # Parse bounding boxes
                pred_bboxes = extract_roi(
                    generated_text, pattern=r"<(\d+)><(\d+)><(\d+)><(\d+)>"
                )
                
                # Convert to list of absolute bbox coordinates
                bboxes = []
                for bbox in pred_bboxes:
                    rel_bbox = [int(coord) for coord in bbox]
                    abs_bbox = self.convert_relative_to_absolute(rel_bbox, image_size)
                    bboxes.append(abs_bbox)
                
                results["detections"][class_name] = bboxes
                
        elif self.task_config == "open_ended":
            # Get instruction data for this image
            instruction_data = self.instruction_map.get(image_path)
            if not instruction_data:
                print(f"Warning: No instruction found for image {image_path}")
                return results
                
            # Add instruction metadata to results
            results.update({
                "instruction": instruction_data.get('instruction'),
                "dataset_type": instruction_data.get('dataset_type'),
                "gt_object": instruction_data.get('objects')
            })

            # Generate and run prompt
            prompt = self.generate_prompt(instruction_data=instruction_data)
            
            # Run Falcon inference
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=8192,
                num_beams=3,
                do_sample=False,
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            print(generated_text)
            # Parse bounding boxes
            pred_bboxes = extract_roi(
                generated_text, pattern=r"<(\d+)><(\d+)><(\d+)><(\d+)>"
            )
            
            # Convert to list of absolute bbox coordinates
            bboxes = []
            for bbox in pred_bboxes:
                rel_bbox = [int(coord) for coord in bbox]
                abs_bbox = self.convert_relative_to_absolute(rel_bbox, image_size)
                bboxes.append(abs_bbox)
            
            results["detections"] = bboxes
        print(results)    
        return results
    
    def save_results(self, image_path: str, results: dict):
        """
        Save detection results to JSON file
        
        :param image_path: Original image path
        :param results: Detection results dictionary
        """
        # Get image name without extension
        image_name = Path(image_path).stem
        output_path = os.path.join(self.output_dir, f"{image_name}.json")
        
        with open(output_path, 'w') as f:
            # 使用紧凑格式但保留缩进（不会产生空行）
            json.dump(results, f)
    
    def run_evaluation(self):
        """Run evaluation on all images in the list"""
        # Read image list
        with open(self.image_list_path) as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        # Process each image
        for image_path in tqdm(image_paths, desc="Processing images"):
            if not os.path.exists(image_path):
                print(f"Warning: Image not found - {image_path}")
                continue
            
            try:
                results = self.process_image(image_path)
                self.save_results(image_path, results)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue

if __name__ == "__main__":
    # Configuration
    CHECKPOINT_PATH = "./Falcon-Single-Instruction-Large"  # Path to Falcon model checkpoint
    IMAGE_LIST = "./datasets/VLAD_Remote/test_image_list.txt"  # Path to text file with image paths
    INSTRUCTIONS_PATH = "./tools/dataset_Remote/instructions_with_objects.json"  # Path to instructions JSON file
    
    # Example usage for fixed classes
    fixed_classes = ["Car", "Bus", "Truck", "Building"]
    fixed_evaluator = FalconBBoxEvaluator(
        CHECKPOINT_PATH,
        IMAGE_LIST, 
        fixed_classes, 
        task_config="fixed"
    )
    fixed_evaluator.run_evaluation()
    
    # Example usage for open classes
    open_evaluator = FalconBBoxEvaluator(
        CHECKPOINT_PATH,
        IMAGE_LIST, 
        [], 
        task_config="open"
    )
    open_evaluator.run_evaluation()
    
    # # Example usage for open-ended task
    # open_ended_evaluator = FalconBBoxEvaluator(
    #     CHECKPOINT_PATH,
    #     IMAGE_LIST, 
    #     [], 
    #     task_config="open_ended",
    #     instructions_path=INSTRUCTIONS_PATH
    # )
    # open_ended_evaluator.run_evaluation()