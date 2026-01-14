from huggingface_hub import snapshot_download
from huggingface_hub import login  # 新增登录函数
import os

# 第一步：登录HF账号（使用你的token）
login(token="")  # 替换为你的实际token

# 模型名称
model_name = "TianHuiLab/Falcon-Single-Instruction-Large"
local_dir = "./falcon"

# 创建目录
os.makedirs(local_dir, exist_ok=True)

# 下载模型（关键修改：添加token和gated参数）
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    token=True,  # 必须启用token验证
    local_dir_use_symlinks=False,
    resume_download=True
)

print("下载完成！")