import os
import zipfile
from tqdm import tqdm
from huggingface_hub import login, snapshot_download

from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token) # token=Your_HuggingFace_Token


def download_hest(patterns, local_dir):
    repo_id = "MahmoodLab/hest"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=patterns,
        max_workers=1
    )
    # Unzip cellvit segmentations if present
    seg_dir = os.path.join(local_dir, "cellvit_seg")
    if os.path.exists(seg_dir):
        print("Unzipping cell vit segmentation...")
        for filename in tqdm([s for s in os.listdir(seg_dir) if s.endswith(".zip")]):
            path_zip = os.path.join(seg_dir, filename)
            with zipfile.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(seg_dir)

data_path = "/storage/hest1k/"   # or whatever root you want
os.makedirs(data_path, exist_ok=True)

# 2. Download the *entire* HEST-1k (~1 TB)
download_hest("*", data_path)
