from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveInt, ConfigDict

# If you already export Stem_models somewhere, import and reuse it instead
StemModelName = Literal["Stem"]   # → replace with Literal[*Stem_models.keys()]

class InferenceConfig(BaseModel):
    # enable post-hoc attributes
    model_config = ConfigDict(extra='allow')  # sorry py wizards, I dont want to modify their code too much

    # ── model ──────────────────────────────────────────────────────
    model: StemModelName
    DiT_num_blocks: PositiveInt
    hidden_size: PositiveInt
    num_heads: PositiveInt

    # ── data / inputs ──────────────────────────────────────────────
    slide_out: str
    gene_list_filename: str
    data_path: Path

    # ── sampling hyper-parameters ──────────────────────────────────
    sample_num_per_cond: PositiveInt = Field(..., description="Samples per input")
    num_sampling_steps: PositiveInt
    seed: int
    sampling_batch_size: PositiveInt

    # ── runtime paths ──────────────────────────────────────────────
    save_path: Optional[Path] = Field(
        ..., description="Where generated samples will be stored"
    )
    ckpt: Path

    # ── device ─────────────────────────────────────────────────────
    device: str = Field("cuda", regex="^(cpu|cuda.*)$")
