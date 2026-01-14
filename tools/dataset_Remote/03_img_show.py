import os
import json
import cv2
from typing import Dict, Optional, List
# from utils.visualize_bboxes import visualize_bboxes
'''
    v0.1.0 2025.06.12 @jialei
    给定图片和标注框（标注框在sharegpt格式的value下，符合json格式）
    将标注框标在图片上
'''
def visualize_bboxes(
    input_image_path: str,
    output_image_path: str,
    json_str: str,
    class_colors: Optional[Dict[str, tuple]] = None
) -> None:
    """

    Args:
        input_image_path: Path to the input image
        output_image_path: Path to save the output image
        json_str: JSON string containing bounding box information
        class_colors: Optional dictionary mapping class names to BGR colors
                     Default is {
                         "small vehicle": (0, 255, 0),
                         "large vehicle": (255, 0, 0),
                         "person": (0, 0, 255)
                     }
    """
    # Set default colors if not provided
    if class_colors is None:
        class_colors = {
            "small vehicle": (0, 255, 0),  # Green
            "large vehicle": (255, 0, 0),  # Red
            "person": (0, 0, 255)         # Blue
        }
    # Read input image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"Could not read image from {input_image_path}")
    # Parse JSON string
    try:
        bboxes = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")
    # Draw each bounding box
    for bbox in bboxes:
        class_name = bbox.get("class", "object")
        bbox_2d = bbox["bbox"]
        obj_id = bbox.get("id", 0)
        # Get color for the class, default to white if class not found
        color = class_colors.get(class_name.lower(), (255, 255, 255))
        # Convert bbox coordinates to integers
        x1, y1, x2, y2 = map(int, bbox_2d)
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        # Create label text
        label = f"{class_name}:{obj_id}"
        # Calculate text size
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        # Draw text background
        cv2.rectangle(
            image, 
            (x1, y1 - text_height - 4), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        # Draw text
        cv2.putText(
            image, 
            label, 
            (x1, y1 - 4), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            1
        )
    # Save output image
    cv2.imwrite(output_image_path, image)
    print(f"Result saved to {output_image_path}")
def visualize_specific_image(
    json_path: str, 
    target_image: str, 
    output_dir: str,
    specific_class: Optional[str] = None
) -> None:
    """
    Visualize bounding boxes for a specific image from JSON annotations in the new format.
    Supports filtering by specific class and handles multiple annotations per image.
    
    Args:
        json_path: Path to JSON annotation file
        target_image: Filename of the specific image to visualize
        output_dir: Directory to save the output visualization
        specific_class: If specified, only visualize this class (e.g., "pedestrian")
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    found = False
    
    # Process each entry in the JSON file
    for entry in data:
        image_path = entry["images"][0]  # Get the first image path
        image_name = os.path.basename(image_path)
        
        # Skip if not the target image
        if image_name != target_image:
            continue
            
        found = True
        
        # Extract class type from the ID (e.g., "pedestrian" from "9999965_00000_d_0000055_pedestrian")
        class_type = entry["id"].split("_")[-1]
        
        # Skip if we're filtering by class and this entry doesn't match
        if specific_class is not None and class_type.lower() != specific_class.lower():
            continue
        
        # Prepare output path
        output_filename = f"{os.path.splitext(image_name)[0]}_{class_type}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        # Convert bounding boxes to the required JSON format
        bboxes = []
        for i, bbox in enumerate(entry["objects"]["bbox"]):
            bboxes.append({
                "id": i + 1,
                "class": class_type,
                "bbox_2d": bbox
            })
        
        # Convert to JSON string
        bbox_json_str = json.dumps(bboxes)
        
        # Visualize bounding boxes
        try:
            visualize_bboxes(
                input_image_path=image_path,
                output_image_path=output_path,
                json_str=bbox_json_str
            )
            print(f"Successfully visualized {class_type} for: {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    if not found:
        print(f"Target image '{target_image}' not found in the JSON file")

# Configuration - modify these as needed
JSON_FILE_PATH = "./results/eval/labels/ms-swift/output/export_v5_11968/fixed/0000205_01665_d_0000200.json"#"./datasets/VLAD_Remote/VLAD_Remote.json" # Update with your JSON path
TARGET_IMAGE = "0000205_01665_d_0000200.jpg"    # The specific image you want to visualize
OUTPUT_DIR = "./results/datasets/remote/gt"
SPECIFIC_CLASS = None  # Set to None to visualize all classes, or specify a class like "pedestrian"

# Run the visualization
if __name__ == "__main__":
    visualize_specific_image(
        json_path=JSON_FILE_PATH,
        target_image=TARGET_IMAGE,
        output_dir=OUTPUT_DIR,
        specific_class=SPECIFIC_CLASS
    )
