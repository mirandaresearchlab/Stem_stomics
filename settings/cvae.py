from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

CVAEModelName = Literal["CVAE"]


class CVAETrainingConfig(BaseModel):
    # enable post-hoc attributes, mirroring TrainingConfig
    model_config = ConfigDict(extra="allow")

    # data
    data_path: Path
    results_dir: Path
    slide_out: str = Field(..., description="Test slide ID")
    folder_list_filename: str
    gene_list_filename: str
    num_aug_ratio: PositiveInt

    # model
    model: CVAEModelName
    hidden_size: PositiveInt
    num_layers: PositiveInt
    latent_dim: PositiveInt
    cond_hidden_size: PositiveInt

    # training
    lr: float
    total_epochs: PositiveInt
    global_batch_size: PositiveInt
    global_seed: int
    num_workers: PositiveInt
    ckpt_every: PositiveInt
    beta: float = Field(1.0, description="KL weight in the beta-weighted ELBO (1.0 = vanilla VAE)")


class CVAEInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    # model
    model: CVAEModelName
    hidden_size: PositiveInt
    num_layers: PositiveInt
    latent_dim: PositiveInt
    cond_hidden_size: PositiveInt

    # data / inputs
    slide_out: str
    gene_list_filename: str
    data_path: Path

    # sampling hyper-parameters
    sample_num_per_cond: PositiveInt = Field(..., description="Samples per input")
    seed: int
    sampling_batch_size: PositiveInt

    # runtime paths
    save_path: Optional[Path] = Field(..., description="Where generated samples will be stored")
    ckpt: Path

    # device
    device: str = Field("cuda", pattern=r"^(cpu|cuda.*)$")
