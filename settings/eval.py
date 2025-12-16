from pathlib import Path
from pydantic import BaseModel, Field, PositiveInt, ConfigDict


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    # data
    data_path: Path
    processed_subdir: str = Field("processed_data", description="Subdirectory under data_path for processed data")
    st_subdir: str = Field("st", description="Subdirectory under data_path for spatial transcriptomics h5ad files")
    slide_out: str
    folder_list_filename: str
    gene_list_filename: str

    # evaluation inputs
    sample_path: Path = Field(..., description="Path to generated samples tensor")
    results_dir: Path = Field(..., description="Directory to save metrics and plots")
    num_rep: PositiveInt = Field(..., description="Number of generated samples per patch")
    num_selected: PositiveInt = Field(..., description="How many samples to average per patch")
    seed: int = 0
