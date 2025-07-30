import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, Dataset

import numpy as np
from copy import deepcopy
from glob import glob
import argparse
import os
import pandas as pd
import random
import anndata

import sys
# sys.path.append("./Stem")
from Stem.models import Stem_models
from Stem.diffusion import create_diffusion
from Stem.train_helper import *

import wandb
from pathlib import Path
from settings.training import TrainingConfig
from utils.config_loader import load_toml_config


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

def ddp_setup(rank, world_size, available_gpus):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(available_gpus[rank])


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
        wandb.watch(self.model, log="all")
        self.ema = deepcopy(model).to(gpu_id)
        requires_grad(self.ema, False)
        self.model = DDP(self.model.to(gpu_id), device_ids=[self.gpu_id])
        self.diffusion = create_diffusion(timestep_respacing="")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                           lr=self.args.lr, weight_decay=0)
        update_ema(self.ema, self.model.module, decay=0)
        self.args.logger.info(f"Rank {rank} - Initializing Trainer... DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

        self.train_steps=0
        self.log_steps=0
        self.running_loss=0

    def _run_batch(self, x, t, modelkwargs):

        loss_dict = self.diffusion.training_losses(self.model, x, t, modelkwargs)
        loss = loss_dict["loss"].mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        update_ema(self.ema, self.model.module)

        self.running_loss += loss.item()
        self.train_steps += 1
        self.log_steps += 1
        if self.log_steps % 500 == 0:
            torch.cuda.synchronize()
            avg_loss = torch.tensor(self.running_loss / self.log_steps, device=x.device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            self.args.logger.info(f"Step={self.train_steps:07d} | Training Loss: {avg_loss:.5f}")
            self.running_loss = 0
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
            x = x.unsqueeze(1).to(self.gpu_id)  # (N, 1, NumGene)
            y = y.to(self.gpu_id)               # (N, NumEmbed)
            t = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device=x.device)
            model_kwargs = dict(y=y)
            self._run_batch(x, t, model_kwargs)

    def _save_checkpoint(self):
        checkpoint = {
                      "model": self.model.module.state_dict(),
                      "ema": self.ema.state_dict(),
                      "opt": self.optimizer.state_dict()
                    }
        checkpoint_path = f"{self.args.checkpoint_dir}/{self.train_steps:07d}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.args.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def train(self, max_epochs: int):
        ##
        self.model.train()
        self.ema.eval()
        ##
        for epoch in range(max_epochs):
            self._run_epoch(epoch)


def assemble_dataset(input_args):
    data_path_str = str(input_args.data_path) + "/"  # convert Path to str for compatibility
    # load & assemble data
    # leave the test slide out
    slidename_lst = list(np.genfromtxt(data_path_str + "processed_data/" + input_args.folder_list_filename, dtype=str))
    for slide_out in input_args.slide_out.split(","):
        slidename_lst.remove(slide_out)
        input_args.logger.info(f"{slide_out} is held out for testing.")
    input_args.logger.info(f"Remaining {len(slidename_lst)} slides: {slidename_lst}")

    # load selected gene list
    selected_genes = list(np.genfromtxt(data_path_str + "processed_data/" + input_args.gene_list_filename, dtype=str))
    input_args.input_gene_size = len(selected_genes)
    input_args.logger.info(f"Selected genes filename: {input_args.gene_list_filename} | len: {len(selected_genes)}")


    # load original patches
    first_slide = True
    all_img_ebd_ori = None
    all_count_mtx_ori = None
    input_args.logger.info("Loading original data...")
    for sni in range(len(slidename_lst)):
        sample_name = slidename_lst[sni]
        test_adata = anndata.read_h5ad(data_path_str + "st/" + sample_name + ".h5ad")
        test_count_mtx = pd.DataFrame(test_adata[:, selected_genes].X.toarray(), 
                                      columns=selected_genes, 
                                      index=[sample_name + "_" + str(i) for i in range(test_adata.shape[0])])
        
        if first_slide:
            all_count_mtx_ori = test_count_mtx
            img_ebd_uni   = torch.load(data_path_str + "processed_data/1spot_uni_ebd/"   + sample_name + "_uni.pt",   map_location="cpu")
            img_ebd_conch = torch.load(data_path_str + "processed_data/1spot_conch_ebd/" + sample_name + "_conch.pt", map_location="cpu")
            all_img_ebd_ori = torch.cat([img_ebd_uni, img_ebd_conch], axis=1)
            input_args.logger.info(f"{sample_name} loaded, count_mtx shape: {all_count_mtx_ori.shape}  | img ebd shape: {all_img_ebd_ori.shape}")
            first_slide = False
            continue
        
        img_ebd_uni   = torch.load(data_path_str + "processed_data/1spot_uni_ebd/"   + sample_name + "_uni.pt",   map_location="cpu")
        img_ebd_conch = torch.load(data_path_str + "processed_data/1spot_conch_ebd/" + sample_name + "_conch.pt", map_location="cpu")
        slide_img_ebd = torch.cat([img_ebd_uni, img_ebd_conch], axis=1)
        all_img_ebd_ori = torch.cat([all_img_ebd_ori, slide_img_ebd], axis=0)
        all_count_mtx_ori = np.concatenate((all_count_mtx_ori, test_count_mtx), axis=0)
        input_args.logger.info(f"{sample_name} loaded, count_mtx shape: {all_count_mtx_ori.shape} | img ebd shape: {all_img_ebd_ori.shape}")
    input_args.cond_size = all_img_ebd_ori.shape[1]
    
    # load augmented patches
    first_slide = True
    all_img_ebd_aug = None
    input_args.logger.info(f"Augmentation data loading...")
    for sni in range(len(slidename_lst)):
        sample_name = slidename_lst[sni]

        if first_slide:
            img_ebd_uni   = torch.load(data_path_str + "processed_data/1spot_uni_ebd_aug/"   + sample_name + "_uni_aug.pt",   map_location="cpu")
            img_ebd_conch = torch.load(data_path_str + "processed_data/1spot_conch_ebd_aug/" + sample_name + "_conch_aug.pt", map_location="cpu")
            all_img_ebd_aug = torch.cat([img_ebd_uni, img_ebd_conch], axis=-1)
            input_args.logger.info(f"With augmentation {sample_name} loaded, img_ebd_mtx shape: {all_img_ebd_aug.shape}, all_img_ebd shape: {all_img_ebd_aug.shape}")
            first_slide = False
            continue
        
        img_ebd_uni   = torch.load(data_path_str + "processed_data/1spot_uni_ebd_aug/"   + sample_name + "_uni_aug.pt",   map_location="cpu")
        img_ebd_conch = torch.load(data_path_str + "processed_data/1spot_conch_ebd_aug/" + sample_name + "_conch_aug.pt", map_location="cpu")
        slide_img_ebd = torch.cat([img_ebd_uni, img_ebd_conch], axis=-1)
        all_img_ebd_aug = torch.cat([all_img_ebd_aug, slide_img_ebd], axis=0)
        input_args.logger.info(f"With augmentation {sample_name} loaded, img_ebd_mtx shape: {slide_img_ebd.shape}, all_img_ebd shape: {all_img_ebd_aug.shape}")
     
    # randomly select augmented patches according to the input augmentation ratio (int)
    num_aug_ratio = input_args.num_aug_ratio
    all_count_mtx_aug = np.repeat(np.copy(all_count_mtx_ori), num_aug_ratio, axis=0)             # generate count matrix for all augmented patches
    selected_img_ebd_aug = torch.zeros((all_count_mtx_aug.shape[0], all_img_ebd_aug.shape[2]))
    for i in range(all_img_ebd_aug.shape[0]):                                                    # randomly select augmented patches
        selected_transpose_idx = np.random.choice(all_img_ebd_aug.shape[1], num_aug_ratio, replace=False)
        selected_img_ebd_aug[i*num_aug_ratio:(i+1)*num_aug_ratio, :] = all_img_ebd_aug[i, selected_transpose_idx, :]

    all_img_ebd = torch.cat([all_img_ebd_ori, selected_img_ebd_aug], axis=0)
    all_count_mtx = np.concatenate((all_count_mtx_ori, all_count_mtx_aug), axis=0)
    input_args.logger.info(f"{num_aug_ratio}:1 augmentation. CONCH+UNI. final count_mtx shape: {all_count_mtx.shape} | final img_ebd shape: {all_img_ebd.shape}")
    
    ################################################
    all_count_mtx_df = pd.DataFrame(all_count_mtx, columns=selected_genes, index=list(range(all_count_mtx.shape[0])))
    # remove the spot with all NAN/zero in count mtx
    all_count_mtx_all_nan_spot_index = all_count_mtx_df.index[all_count_mtx_df.isnull().all(axis=1)]
    all_count_mtx_all_zero_spot_index = all_count_mtx_df.index[all_count_mtx_df.sum(axis=1) == 0]
    input_args.logger.info(f"All NAN spot index: {all_count_mtx_all_nan_spot_index}")
    input_args.logger.info(f"All zero spot index: {all_count_mtx_all_zero_spot_index}")
    spot_idx_to_remove = list(set(all_count_mtx_all_nan_spot_index) | set(all_count_mtx_all_zero_spot_index))
    spot_idx_to_keep = list(set(all_count_mtx_df.index) - set(spot_idx_to_remove))
    all_count_mtx = all_count_mtx_df.loc[spot_idx_to_keep, :]
    all_img_ebd = all_img_ebd[spot_idx_to_keep, :]
    input_args.logger.info(f"After exclude rows with all nan/zeros: {all_count_mtx.shape}, {all_img_ebd.shape}")
    # only normalized by log2(+1)
    all_count_mtx_selected_genes = np.log2(all_count_mtx.loc[:, selected_genes] + 1).copy()
    input_args.logger.info(f"Selected genes count matrix shape: {all_count_mtx_selected_genes.shape}" )
    all_img_ebd.requires_grad_(False)
    alldataset = CustomDataset(torch.from_numpy(all_count_mtx_selected_genes.values).float(), 
                               all_img_ebd.float())    
    return alldataset, input_args


def load_train_objs(args):
    train_set, args = assemble_dataset(args)
    model = Stem_models[args.model](
        input_size=args.input_gene_size,
        depth= args.DiT_num_blocks,
        hidden_size=args.hidden_size, 
        num_heads=args.num_heads, 
        label_size=args.cond_size,
    )
    args.logger.info(f"Dataset contains {len(train_set):,} images ({args.data_path})")
    return train_set, model, args


def prepare_dataloader(args, dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset,
                                   shuffle=True,
                                   seed=args.global_seed),
        num_workers=args.num_workers,
        drop_last=True,
    )

def main(world_size: int, 
         available_gpus: list,
         cfg: TrainingConfig):
    
    # Set up DDP
    dist.init_process_group(backend="nccl", world_size=world_size)
    rank = dist.get_rank()
    device = available_gpus[rank]
    seed = cfg.global_seed * dist.get_world_size() + rank
    print("Rank: ", rank, " | Device: ", device, " | Seed: ", seed)
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # set up output folder and logger
    if rank == 0:
        print("Rank 0 mkdir & set up logger...")
        # mkdir for logs and checkpoints
        os.makedirs(cfg.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{cfg.results_dir}/*"))
        cfg.experiment_dir = f"{cfg.results_dir}/{experiment_index:03d}"  # Create an experiment folder
        cfg.checkpoint_dir = f"{cfg.experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(f"{cfg.experiment_dir}/samples", exist_ok=True)      # Store sampling results
        cfg.logger = create_logger(cfg.experiment_dir)
        cfg.logger.info(f"Experiment directory created at {cfg.experiment_dir}")
    else:
        cfg.logger=create_logger(None)
    cfg.logger.info(f"Rank: {rank} | Device: {device} | Seed: {seed}")
    
    # set up training objects
    dataset, model, args = load_train_objs(cfg)
    cfg.logger.info(f"Dataset, model, and args finished loading.")
    train_data = prepare_dataloader(args, dataset, 
                                    int(args.global_batch_size // dist.get_world_size()))
    cfg.logger.info(f"Dataloader finished loading.")
    trainer = Trainer(model, train_data, 
                      rank, int(device.split(":")[-1]), 
                      args)
    cfg.logger.info(f"Trainer finished loading.")
    cfg.logger.info(f"Starting...")
    trainer.train(args.total_epochs)
    destroy_process_group()


def parse_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Train DiT model using a TOML configuration file"
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path to the training TOML config (required)",
    )
    return parser.parse_args().config



def _cli_entrypoint():
    cfg_path: Path = parse_args()
    cfg: TrainingConfig = load_toml_config(cfg_path, TrainingConfig)
    print("▶ loaded config:\n", cfg)

    available_gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    print("Available GPUs: ", available_gpus)
    wandb.init(project="stomics", config=cfg)
    main(cfg.num_workers, available_gpus, cfg)


if __name__ == "__main__":
    _cli_entrypoint()
