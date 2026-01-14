from huggingface_hub import snapshot_download

# 下载数据集到指定目录
snapshot_download(
    repo_id="xiang709/VRSBench",
    repo_type="dataset",
    local_dir="./datasets/VRSBench",  # 保存到当前目录下的datasets/VRSBench
    local_dir_use_symlinks=False,     # 避免符号链接
    # token="你的HF_TOKEN"              # 如果需要访问私有数据集
)