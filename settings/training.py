from pathlib import Path
from pydantic import BaseModel, Field, PositiveInt, ConfigDict

class TrainingConfig(BaseModel):
    # enable post-hoc attributes
    model_config = ConfigDict(extra='allow')  # sorry py wizards, I dont want to modify their code too much

    # -------- data ----------
    data_path: Path
    results_dir: Path
    slide_out: str                = Field(..., description="Test slide ID")
    folder_list_filename: str
    gene_list_filename: str
    num_aug_ratio: PositiveInt
    # -------- model ----------
    model: str
    DiT_num_blocks: PositiveInt
    hidden_size: PositiveInt
    num_heads: PositiveInt
    # -------- training -------
    lr: float
    total_epochs: PositiveInt
    global_batch_size: PositiveInt
    global_seed: int
    num_workers: PositiveInt
    ckpt_every: PositiveInt