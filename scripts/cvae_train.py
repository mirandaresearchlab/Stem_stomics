"""
Conditional-VAE training script.
This is a minimal copy of scripts/fm_train.py with the following changes:
- swap the flow-matching objective for a beta-weighted ELBO (Stem.vae.cvae_loss)
- the model is a plain MLP CVAE (Stem.vae.CVAE_models), not the DiT backbone
- x is kept as (N, NumGene) — the CVAE has no per-gene token / channel axis
"""

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset

import numpy as np
from copy import deepcopy
from glob import glob
import argparse
import os
import pandas as pd
import random
import anndata

from Stem.vae import CVAE_models, cvae_loss
from Stem.train_helper import *

import wandb
from pathlib import Path
from settings.cvae import CVAETrainingConfig
from utils.config_loader import load_toml_config


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        rank: int,
        gpu_id: int,
        model_args: argparse.Namespace,
    ) -> None:
        self.rank = rank
        self.gpu_id = gpu_id
        self.train_data = train_data
        self.args = model_args

        self.model = model
        wandb.watch(self.model, log="all", log_freq=100)
        self.ema = deepcopy(model).to(gpu_id)
        requires_grad(self.ema, False)
        self.model = DDP(self.model.to(gpu_id), device_ids=[self.gpu_id])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=0)
        update_ema(self.ema, self.model.module, decay=0)
        self.args.logger.info(
            f"Rank {rank} - Initializing CVAE Trainer... CVAE Parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

        self.train_steps = 0
        self.log_steps = 0
        self.running_loss = 0
        self.running_recon = 0
        self.running_kl = 0

    def _run_batch(self, x, y):
        x_recon, mu, logvar = self.model(x, y)
        loss_dict = cvae_loss(x, x_recon, mu, logvar, beta=self.args.beta)
        loss = loss_dict["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        if self.args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.optimizer.step()
        update_ema(self.ema, self.model.module)

        self.running_loss += loss.item()
        self.running_recon += loss_dict["recon"].item()
        self.running_kl += loss_dict["kl"].item()
        self.train_steps += 1
        self.log_steps += 1
        if self.log_steps % 50 == 0:
            torch.cuda.synchronize()
            avg_loss = torch.tensor(self.running_loss / self.log_steps, device=x.device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            avg_recon = self.running_recon / self.log_steps
            avg_kl = self.running_kl / self.log_steps

            if self.rank == 0:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/recon": avg_recon,
                    "train/kl": avg_kl,
                    "train/beta": self.args.beta,
                    "step": self.train_steps,
                })

            self.args.logger.info(
                f"Step={self.train_steps:07d} | Loss: {avg_loss:.5f} | Recon: {avg_recon:.5f} | KL: {avg_kl:.5f}"
            )
            self.running_loss = 0
            self.running_recon = 0
            self.running_kl = 0
            self.log_steps = 0

        if self.train_steps % self.args.ckpt_every == 0 and self.train_steps > 0:
            if self.rank == 0:
                self._save_checkpoint()
            dist.barrier()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)

        for x, y in self.train_data:
            x = x.to(self.gpu_id)  # (N, NumGene)
            y = y.to(self.gpu_id)  # (N, NumEmbed)
            self._run_batch(x, y)

    def _save_checkpoint(self):
        checkpoint = {
            "model": self.model.module.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.optimizer.state_dict(),
        }
        checkpoint_path = f"{self.args.checkpoint_dir}/{self.train_steps:07d}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.args.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def train(self, max_epochs: int):
        self.model.train()
        self.ema.eval()
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
        if self.rank == 0:
            self._save_checkpoint()  # final save


def assemble_dataset(input_args):
    data_path_str = str(input_args.data_path) + "/"  # convert Path to str for compatibility
    # load & assemble data
    slidename_lst = list(
        np.genfromtxt(data_path_str + "processed_data/" + input_args.folder_list_filename, dtype=str)
    )
    for slide_out in input_args.slide_out.split(","):
        slidename_lst.remove(slide_out)
        input_args.logger.info(f"{slide_out} is held out for testing.")
    input_args.logger.info(f"Remaining {len(slidename_lst)} slides: {slidename_lst}")

    selected_genes = list(
        np.genfromtxt(data_path_str + "processed_data/" + input_args.gene_list_filename, dtype=str)
    )
    input_args.input_gene_size = len(selected_genes)
    input_args.logger.info(
        f"Selected genes filename: {input_args.gene_list_filename} | len: {len(selected_genes)}"
    )

    # load original patches
    first_slide = True
    all_img_ebd_ori = None
    all_count_mtx_ori = None
    input_args.logger.info("Loading original data...")
    for sni in range(len(slidename_lst)):
        sample_name = slidename_lst[sni]
        try:
            test_adata = anndata.read_h5ad(data_path_str + "st/" + sample_name + ".h5ad")
        except Exception:
            test_adata = anndata.read_h5ad(data_path_str + "../st/" + sample_name + ".h5ad")

        test_count_mtx = pd.DataFrame(
            test_adata[:, selected_genes].X.toarray(),
            columns=selected_genes,
            index=[sample_name + "_" + str(i) for i in range(test_adata.shape[0])],
        )

        if first_slide:
            all_count_mtx_ori = test_count_mtx
            img_ebd_uni = torch.load(
                data_path_str + "processed_data/1spot_uni_ebd/" + sample_name + "_uni.pt",
                map_location="cpu",
            )
            img_ebd_conch = torch.load(
                data_path_str + "processed_data/1spot_conch_ebd/" + sample_name + "_conch.pt",
                map_location="cpu",
            )
            all_img_ebd_ori = torch.cat([img_ebd_uni, img_ebd_conch], axis=1)
            input_args.logger.info(
                f"{sample_name} loaded, count_mtx shape: {all_count_mtx_ori.shape}  | img ebd shape: {all_img_ebd_ori.shape}"
            )
            first_slide = False
            continue

        img_ebd_uni = torch.load(
            data_path_str + "processed_data/1spot_uni_ebd/" + sample_name + "_uni.pt",
            map_location="cpu",
        )
        img_ebd_conch = torch.load(
            data_path_str + "processed_data/1spot_conch_ebd/" + sample_name + "_conch.pt",
            map_location="cpu",
        )
        slide_img_ebd = torch.cat([img_ebd_uni, img_ebd_conch], axis=1)
        all_img_ebd_ori = torch.cat([all_img_ebd_ori, slide_img_ebd], axis=0)
        all_count_mtx_ori = np.concatenate((all_count_mtx_ori, test_count_mtx), axis=0)
        input_args.logger.info(
            f"{sample_name} loaded, count_mtx shape: {all_count_mtx_ori.shape} | img ebd shape: {all_img_ebd_ori.shape}"
        )
    input_args.cond_size = all_img_ebd_ori.shape[1]

    # load augmented patches
    first_slide = True
    all_img_ebd_aug = None
    input_args.logger.info("Augmentation data loading...")
    for sni in range(len(slidename_lst)):
        sample_name = slidename_lst[sni]

        if first_slide:
            img_ebd_uni = torch.load(
                data_path_str + "processed_data/1spot_uni_ebd_aug/" + sample_name + "_uni_aug.pt",
                map_location="cpu",
            )
            img_ebd_conch = torch.load(
                data_path_str + "processed_data/1spot_conch_ebd_aug/" + sample_name + "_conch_aug.pt",
                map_location="cpu",
            )
            all_img_ebd_aug = torch.cat([img_ebd_uni, img_ebd_conch], axis=-1)
            input_args.logger.info(
                f"With augmentation {sample_name} loaded, img_ebd_mtx shape: {all_img_ebd_aug.shape}, all_img_ebd shape: {all_img_ebd_aug.shape}"
            )
            first_slide = False
            continue

        img_ebd_uni = torch.load(
            data_path_str + "processed_data/1spot_uni_ebd_aug/" + sample_name + "_uni_aug.pt",
            map_location="cpu",
        )
        img_ebd_conch = torch.load(
            data_path_str + "processed_data/1spot_conch_ebd_aug/" + sample_name + "_conch_aug.pt",
            map_location="cpu",
        )
        slide_img_ebd = torch.cat([img_ebd_uni, img_ebd_conch], axis=-1)
        all_img_ebd_aug = torch.cat([all_img_ebd_aug, slide_img_ebd], axis=0)
        input_args.logger.info(
            f"With augmentation {sample_name} loaded, img_ebd_mtx shape: {slide_img_ebd.shape}, all_img_ebd shape: {all_img_ebd_aug.shape}"
        )

    num_aug_ratio = input_args.num_aug_ratio
    all_count_mtx_aug = np.repeat(np.copy(all_count_mtx_ori), num_aug_ratio, axis=0)
    selected_img_ebd_aug = torch.zeros((all_count_mtx_aug.shape[0], all_img_ebd_aug.shape[2]))
    for i in range(all_img_ebd_aug.shape[0]):
        selected_transpose_idx = np.random.choice(all_img_ebd_aug.shape[1], num_aug_ratio, replace=False)
        selected_img_ebd_aug[i * num_aug_ratio : (i + 1) * num_aug_ratio, :] = all_img_ebd_aug[
            i, selected_transpose_idx, :
        ]

    all_img_ebd = torch.cat([all_img_ebd_ori, selected_img_ebd_aug], axis=0)
    all_count_mtx = np.concatenate((all_count_mtx_ori, all_count_mtx_aug), axis=0)
    input_args.logger.info(
        f"{num_aug_ratio}:1 augmentation. CONCH+UNI. final count_mtx shape: {all_count_mtx.shape} | final img_ebd shape: {all_img_ebd.shape}"
    )

    all_count_mtx_df = pd.DataFrame(all_count_mtx, columns=selected_genes, index=list(range(all_count_mtx.shape[0])))
    all_count_mtx_all_nan_spot_index = all_count_mtx_df.index[all_count_mtx_df.isnull().all(axis=1)]
    all_count_mtx_all_zero_spot_index = all_count_mtx_df.index[all_count_mtx_df.sum(axis=1) == 0]
    input_args.logger.info(f"All NAN spot index: {all_count_mtx_all_nan_spot_index}")
    input_args.logger.info(f"All zero spot index: {all_count_mtx_all_zero_spot_index}")
    spot_idx_to_remove = list(set(all_count_mtx_all_nan_spot_index) | set(all_count_mtx_all_zero_spot_index))
    spot_idx_to_keep = list(set(all_count_mtx_df.index) - set(spot_idx_to_remove))
    all_count_mtx = all_count_mtx_df.loc[spot_idx_to_keep, :]
    all_img_ebd = all_img_ebd[spot_idx_to_keep, :]
    input_args.logger.info(f"After exclude rows with all nan/zeros: {all_count_mtx.shape}, {all_img_ebd.shape}")
    all_count_mtx_selected_genes = np.log2(all_count_mtx.loc[:, selected_genes] + 1).copy()
    input_args.logger.info(f"Selected genes count matrix shape: {all_count_mtx_selected_genes.shape}")
    all_img_ebd.requires_grad_(False)
    alldataset = CustomDataset(torch.from_numpy(all_count_mtx_selected_genes.values).float(), all_img_ebd.float())
    return alldataset, input_args


def load_train_objs(args):
    train_set, args = assemble_dataset(args)
    model = CVAE_models[args.model](
        input_size=args.input_gene_size,
        label_size=args.cond_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        cond_hidden_size=args.cond_hidden_size,
    )
    args.logger.info(f"Dataset contains {len(train_set):,} images ({args.data_path})")
    return train_set, model, args


def prepare_dataloader(args, dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle=True, seed=args.global_seed),
        num_workers=args.num_workers,
        drop_last=True,
    )


def main(world_size: int, available_gpus: list, cfg: CVAETrainingConfig):
    dist.init_process_group(backend="nccl", world_size=world_size)
    rank = dist.get_rank()
    device = available_gpus[rank]
    seed = cfg.global_seed * dist.get_world_size() + rank
    print("Rank: ", rank, " | Device: ", device, " | Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    if rank == 0:
        print("Rank 0 mkdir & set up logger...")
        os.makedirs(cfg.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{cfg.results_dir}/*"))
        cfg.experiment_dir = f"{cfg.results_dir}/{experiment_index:03d}"
        cfg.checkpoint_dir = f"{cfg.experiment_dir}/checkpoints"
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(f"{cfg.experiment_dir}/samples", exist_ok=True)
        cfg.logger = create_logger(cfg.experiment_dir)
        cfg.logger.info(f"Experiment directory created at {cfg.experiment_dir}")
    else:
        cfg.logger = create_logger(None)
    cfg.logger.info(f"Rank: {rank} | Device: {device} | Seed: {seed}")
    cfg.logger.info(f"CVAE settings — beta: {cfg.beta}, latent_dim: {cfg.latent_dim}")

    dataset, model, args = load_train_objs(cfg)
    cfg.logger.info(f"Dataset, model, and args finished loading.")
    train_data = prepare_dataloader(args, dataset, int(args.global_batch_size // dist.get_world_size()))
    cfg.logger.info(f"Dataloader finished loading.")
    trainer = Trainer(model, train_data, rank, int(device.split(":")[-1]), args)
    cfg.logger.info("Trainer finished loading.")
    cfg.logger.info("Starting...")
    trainer.train(args.total_epochs)
    dist.destroy_process_group()


def parse_args() -> Path:
    parser = argparse.ArgumentParser(description="Train a conditional VAE using a TOML config")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path to the training TOML config (required)",
    )
    return parser.parse_args().config


def _cli_entrypoint():
    cfg_path: Path = parse_args()
    cfg: CVAETrainingConfig = load_toml_config(cfg_path, CVAETrainingConfig, ["model", "data", "training"])

    rank = int(os.environ.get("RANK", 0))
    mode = "online" if rank == 0 else "disabled"
    wandb.init(project="stomics", config=cfg, mode=mode)

    available_gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    main(cfg.num_workers, available_gpus, cfg)


if __name__ == "__main__":
    _cli_entrypoint()
