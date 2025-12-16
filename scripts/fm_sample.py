import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader, Dataset

from Stem.models import Stem_models
from Stem.fm import create_flow_matching
import argparse
import numpy as np
import os

from pathlib import Path

from settings.inference import InferenceConfig
from utils.config_loader import load_toml_config


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def find_model(model_name, device=""):
    assert os.path.isfile(model_name), f"Could not find checkpoint at {model_name}"
    if device == "":
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(model_name, map_location=device)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    return checkpoint


def main(cfg: InferenceConfig):
    torch.manual_seed(cfg.seed)
    torch.set_grad_enabled(False)
    device = cfg.device

    model = Stem_models[cfg.model](
        input_size=cfg.input_gene_size,
        depth=cfg.DiT_num_blocks,
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_heads,
        label_size=cfg.cond_size,
        learn_sigma=False,  # flow matching checkpoints output a single velocity channel
    )

    ckpt_path = cfg.ckpt
    state_dict = find_model(ckpt_path, device=cfg.device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    time_scale = getattr(cfg, "time_scale", 1.0)
    time_eps = getattr(cfg, "time_eps", 1e-3)
    flow_matching = create_flow_matching(time_eps=time_eps, time_scale=time_scale)

    loader = DataLoader(cfg.dataset, batch_size=cfg.sampling_batch_size, shuffle=False)
    all_samples = None
    first_batch = True
    for i, (_, y) in enumerate(loader):
        y = y.to(device)
        model_kwargs = dict(y=y)
        samples = flow_matching.sample_euler(
            model.forward,
            (y.shape[0], 1, cfg.input_gene_size),
            num_steps=cfg.num_sampling_steps,
            model_kwargs=model_kwargs,
            device=device,
            progress=True,
        )
        if first_batch:
            all_samples = samples.detach().cpu()
            first_batch = False
        else:
            all_samples = torch.cat((all_samples, samples.detach().cpu()), dim=0)
        print(str(i) + "/" + str(len(loader)) + " DONE")

    cfg.save_path.mkdir(parents=True, exist_ok=True)
    save_path_append = f"fm_generated_samples_{cfg.ckpt.stem}_{cfg.sample_num_per_cond}sample.pt"
    torch.save(all_samples, cfg.save_path / save_path_append)


def parse_args() -> Path:
    parser = argparse.ArgumentParser(description="Run deterministic gene prediction with a TOML inference config")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=Path,
        metavar="FILE",
        help="Path to inference TOML config (required)",
    )
    return parser.parse_args().config


def _cli_entrypoint():
    cfg_path = parse_args()
    cfg: InferenceConfig = load_toml_config(
        cfg_path, InferenceConfig, sections_to_flatten=["model", "data", "sampling", "paths", "flow"]
    )
    print("▶ loaded inference config:\n", cfg)

    data_path = cfg.data_path
    img_ebd_uni = torch.load(data_path / f"processed_data/1spot_uni_ebd/{cfg.slide_out}_uni.pt")
    img_ebd_conch = torch.load(data_path / f"processed_data/1spot_conch_ebd/{cfg.slide_out}_conch.pt")
    all_img_ebd = torch.cat([img_ebd_uni, img_ebd_conch], dim=1)
    cfg.raw_cond = all_img_ebd
    cfg.cond_size = all_img_ebd.shape[1]

    print("Image patches shape: ", cfg.raw_cond.shape)
    cfg.cond = torch.zeros_like(cfg.raw_cond.repeat((cfg.sample_num_per_cond, 1)))
    print("Total number of samples to generate: ", cfg.cond.shape)
    for i in range(cfg.sample_num_per_cond):
        cfg.cond[i::cfg.sample_num_per_cond] = cfg.raw_cond.clone()

    selected_genes = np.genfromtxt(data_path / f"processed_data/{cfg.gene_list_filename}", dtype=str)
    print("Selected genes are in file - ", cfg.gene_list_filename)
    cfg.input_gene_size = len(selected_genes)

    cfg.dataset = CustomDataset(cfg.cond, cfg.cond)
    print(len(cfg.dataset))

    main(cfg)


if __name__ == "__main__":
    _cli_entrypoint()
