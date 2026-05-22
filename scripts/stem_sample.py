import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader, Dataset

from Stem.models import Stem_models
from Stem.diffusion import create_diffusion
import argparse
import numpy as np
import os
import time

from pathlib import Path

from settings.inference import InferenceConfig
from utils.config_loader import load_toml_config


def write_timing(cfg, sampling_seconds: float, num_rows: int):
    """Write the wall-clock sampling time (and run parameters) to timing.txt."""
    m, s = divmod(sampling_seconds, 60)
    lines = [
        f"sampling_seconds: {sampling_seconds:.3f}",
        f"sampling_time: {int(m)}m{s:06.3f}s",
        f"num_sampling_steps: {cfg.num_sampling_steps}",
        f"device: {cfg.device}",
        f"sampling_batch_size: {cfg.sampling_batch_size}",
        f"num_conditioning_rows: {num_rows}",
    ]
    (cfg.save_path / "timing.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]



def find_model(model_name, device=""):
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    if device == "":
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(model_name, map_location=device)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    return checkpoint


def main(cfg: InferenceConfig):
    # Setup PyTorch:
    torch.manual_seed(cfg.seed)
    torch.set_grad_enabled(False)
    device = cfg.device

    model = Stem_models[cfg.model](
        input_size=cfg.input_gene_size,
        depth= cfg.DiT_num_blocks,
        hidden_size=cfg.hidden_size, 
        num_heads=cfg.num_heads, 
        label_size=cfg.cond_size,
    )   
    
    ckpt_path = cfg.ckpt
    state_dict = find_model(ckpt_path, device=cfg.device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    diffusion = create_diffusion(str(cfg.num_sampling_steps))

    loader = DataLoader(cfg.dataset, batch_size=cfg.sampling_batch_size, shuffle=False)
    all_samples = None
    first_batch = True
    i = 0
    sampling_start = time.perf_counter()
    for _, y in loader:
        y = y.to(device)
        z = torch.randn(y.shape[0], 1, cfg.input_gene_size, device=device)
        model_kwargs = dict(y=y)
        samples = diffusion.p_sample_loop(
            model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        if first_batch:
            all_samples = samples.detach().cpu()
            first_batch = False
        else:
            all_samples = torch.cat((all_samples, samples.detach().cpu()), dim=0)
        print(str(i) + "/" + str(len(loader)) + " DONE")
        i += 1
    if isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.synchronize()  # ensure all GPU work finished before stopping the timer
    sampling_seconds = time.perf_counter() - sampling_start

    cfg.save_path.mkdir(parents=True, exist_ok=True)
    save_path_append = f"generated_samples_{cfg.ckpt.stem}_{cfg.sample_num_per_cond}sample.pt"
    torch.save(all_samples, cfg.save_path / save_path_append)
    write_timing(cfg, sampling_seconds, all_samples.shape[0])


def parse_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Run image-gene sampling with a TOML inference config"
    )
    parser.add_argument(
        "-c", "--config", required=True, type=Path, metavar="FILE",
        help="Path to inference TOML config (required)"
    )
    return parser.parse_args().config

def _cli_entrypoint():
    cfg_path = parse_args()
    cfg: InferenceConfig = load_toml_config(
        cfg_path, InferenceConfig, sections_to_flatten=["model", "data", "sampling", "paths"])
    print("▶ loaded inference config:\n", cfg)

    # load image patches
    data_path = cfg.data_path
    img_ebd_uni   = torch.load(data_path / f"processed_data/1spot_uni_ebd/{cfg.slide_out}_uni.pt")
    img_ebd_conch = torch.load(data_path / f"processed_data/1spot_conch_ebd/{cfg.slide_out}_conch.pt")
    all_img_ebd = torch.cat([img_ebd_uni, img_ebd_conch], dim=1)
    cfg.raw_cond = all_img_ebd
    cfg.cond_size = all_img_ebd.shape[1]

    # create condition matrix
    print("Image patches shape: ", cfg.raw_cond.shape)
    cfg.cond = torch.zeros_like(cfg.raw_cond.repeat((cfg.sample_num_per_cond, 1)))
    print("Total number of samples to generate: ", cfg.cond.shape)
    for i in range(cfg.sample_num_per_cond):
        cfg.cond[i::cfg.sample_num_per_cond] = cfg.raw_cond.clone()

    # load gene list
    selected_genes = np.genfromtxt(data_path / f"processed_data/{cfg.gene_list_filename}", dtype=str)
    print("Selected genes are in file - ", cfg.gene_list_filename)
    cfg.input_gene_size = len(selected_genes)

    # create dataset
    cfg.dataset = CustomDataset(cfg.cond, cfg.cond)
    print(len(cfg.dataset))
    
    main(cfg)

if __name__ == "__main__":
    _cli_entrypoint()
