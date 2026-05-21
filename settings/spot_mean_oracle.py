from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class SpotMeanOracleConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    # data
    data_path: Path
    processed_subdir: str = Field("processed_data", description="Subdirectory under data_path for processed data")
    st_subdir: str = Field("st", description="Subdirectory (relative to data_path/..) for h5ad files")
    slide_out: str
    gene_list_filename: str

    # output
    save_dir: Path = Field(..., description="Directory to write the oracle prediction tensor into")
    num_rep: PositiveInt = Field(
        1,
        description=(
            "Identical replicates per spot in the output tensor. The oracle is deterministic, "
            "so 1 is the natural choice; the eval config must match this value."
        ),
    )
