import json

# 读取原始文件
with open('VRSBench_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取前500行
top_500 = data[:500]

# 保存为新文件
with open('VRSBench_train_top500.json', 'w', encoding='utf-8') as f:
    json.dump(top_500, f, ensure_ascii=False, indent=4)

print("前500行已成功保存为 VRSBench_train_top500.json")