import json
from pathlib import Path

# === 配置路径 ===
BASE_DIR = Path("/home/zirui/.cursor-server/Qwen2.5-VL-FT-Remote")
VLAD_INPUT = BASE_DIR / "datasets/VLAD_Remote/VLAD_Remote_train.json"
VLAD_OUTPUT = BASE_DIR / "datasets/VLAD_Remote/VLAD_Remote_train_abs.json"
VLAD_IMG_ROOT = BASE_DIR / "datasets/VLAD_Remote"

def id_to_filename(id_str: str):
    """
    将 id 转换为图片文件名。
    例：
      '1428_768_1572_2382_Building' → '1428_768_1572_2382.jpg'
      '1234_567_890_111_Car' → '1234_567_890_111.jpg'
    """
    parts = id_str.split("_")
    if len(parts) >= 5:
        stem = "_".join(parts[:4])
    else:
        stem = id_str
    return stem + ".jpg"

def build_image_index():
    """建立文件名索引：stem -> 绝对路径"""
    print("[INFO] 正在扫描 VLAD_Remote 图片目录...")
    index = {}
    for p in VLAD_IMG_ROOT.rglob("*.*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            index[p.stem] = str(p.resolve())
    print(f"[INFO] 已索引 {len(index)} 张图片。")
    return index

def main():
    img_index = build_image_index()

    data = json.load(open(VLAD_INPUT, "r", encoding="utf-8"))
    new_data = []
    missing = 0

    for item in data:
        id_str = item["id"]
        filename = id_to_filename(id_str)
        stem = Path(filename).stem

        abs_path = None
        if stem in img_index:
            abs_path = img_index[stem]
        else:
            # 尝试模糊匹配（以数字部分为关键）
            candidates = [v for k, v in img_index.items() if stem in k]
            if candidates:
                abs_path = candidates[0]
            else:
                missing += 1
                continue

        new_item = {
            "id": id_str,
            "image": abs_path,
            "conversations": item.get("messages", [])
        }
        new_data.append(new_item)

    with open(VLAD_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 转换完成：共 {len(new_data)} 条，缺失 {missing} 条。")
    print(f"输出文件：{VLAD_OUTPUT}")

if __name__ == "__main__":
    main()
