import json
import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib
matplotlib.use('Agg') 
'''
    v0.1.0 2025.06.12 @jialei
    统计数据集中：
    1.（csv）总共有多少图片，多少类别，总共多少个对象
    2.（csv）每一个类别有多少对象，多少图片，平均每个图片包含对象数，单图最多包含对象数
    3. (直方图）包含类别的图像数（前20类）
    4. (直方图）包含类别的对象数（前20类）
'''

def analyze_json_data(json_path: str, output_dir: str):
    """
    Analyze object detection data from JSON file in the new format and generate statistics
    
    Args:
        json_path: Path to JSON file
        output_dir: Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Initialize statistics data structures
    class_stats = defaultdict(int)  # Total objects per class
    image_class_stats = defaultdict(set)  # Which images contain each class
    image_stats = defaultdict(lambda: defaultdict(int))  # Objects per class per image
    class_per_image_dist = defaultdict(list)  # Distribution of objects per class per image
    unique_images = set()  # Track all unique images
    
    # Process all entries
    for entry in data:
        image_path = entry["images"][0]  # Get the first image path
        image_name = os.path.basename(image_path)
        unique_images.add(image_name)
        
        # Extract class name from ID
        class_name = entry["id"].split("_")[-1].replace(" ", "_")
        
        # Get bounding boxes directly from objects field
        bboxes = entry["objects"]["bbox"]
        
        # Update statistics
        num_objects = len(bboxes)
        class_stats[class_name] += num_objects
        image_class_stats[class_name].add(image_name)
        image_stats[image_name][class_name] += num_objects
        class_per_image_dist[class_name].append(num_objects)
    
    # 1. Overall statistics
    total_images = len(unique_images)
    total_classes = len(class_stats)
    total_objects = sum(class_stats.values())
    
    print(f"Overall Statistics:")
    print(f"- Total images: {total_images}")
    print(f"- Total classes: {total_classes}")
    print(f"- Total objects: {total_objects}")
    
    # Save overall statistics to CSV
    overall_csv_path = os.path.join(output_dir, "overall_stats.csv")
    with open(overall_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Class", "Total Objects", "Images with Class", "Avg per Image", "Max per Image"])
        for class_name, count in sorted(class_stats.items(), key=lambda x: -x[1]):
            img_count = len(image_class_stats[class_name])
            avg_per_img = count / img_count if img_count > 0 else 0
            max_per_img = max(class_per_image_dist[class_name]) if class_per_image_dist[class_name] else 0
            writer.writerow([class_name, count, img_count, f"{avg_per_img:.2f}", max_per_img])
    
    # 2. Distribution statistics
    # Objects per class per image
    per_image_stats = []
    for image_name, class_counts in image_stats.items():
        for class_name, count in class_counts.items():
            per_image_stats.append({
                "image": image_name,
                "class": class_name,
                "count": count
            })
    
    # Save distribution statistics to CSV
    distribution_csv_path = os.path.join(output_dir, "distribution_stats.csv")
    with open(distribution_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image", "Class", "Object Count"])
        for stat in per_image_stats:
            writer.writerow([stat["image"], stat["class"], stat["count"]])
    
    # Visualizations
    # 1. Total objects per class (top 20)
    top_classes = sorted(class_stats.items(), key=lambda x: -x[1])[:20]
    plt.figure(figsize=(15, 8))
    plt.bar([x[0] for x in top_classes], [x[1] for x in top_classes])
    plt.title('Total Objects per Class (Top 20)')
    plt.xlabel('Class')
    plt.ylabel('Object Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_object_count.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Images containing each class (top 20)
    class_image_count = {k: len(v) for k, v in image_class_stats.items()}
    top_classes_img = sorted(class_image_count.items(), key=lambda x: -x[1])[:20]
    plt.figure(figsize=(15, 8))
    plt.bar([x[0] for x in top_classes_img], [x[1] for x in top_classes_img])
    plt.title('Images Containing Each Class (Top 20)')
    plt.xlabel('Class')
    plt.ylabel('Image Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_image_count.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Object distribution per class (top 20)
    top_classes_box = sorted(class_per_image_dist.items(), key=lambda x: -len(x[1]))[:20]
    plt.figure(figsize=(15, 8))
    plt.boxplot([x[1] for x in top_classes_box], labels=[x[0] for x in top_classes_box])
    plt.title('Object Distribution per Class (Top 20)')
    plt.xlabel('Class')
    plt.ylabel('Object Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'object_distribution_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis completed. Results saved to: {output_dir}")


# Example usage
if __name__ == "__main__":
    json_path = "./datasets/VLAD_Remote/VLAD_Remote.json"  # Replace with your JSON file path
    output_dir = "./results/datasets/remote/analysis_results"  # Output directory
    
    analyze_json_data(json_path, output_dir)