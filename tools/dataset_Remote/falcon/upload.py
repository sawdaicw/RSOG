from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./falcon/Falcon-Single-Instruction-Large",
    repo_id="juanjuanjuanbing/falcon",
    repo_type="model"
)