from pathlib import Path
import tomllib
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

def load_toml_config(path: str | Path, model_class: Type[T]) -> T:
    """Read TOML → flatten tables → return a validated Pydantic object.
    
    Args:
        path: Path to the TOML file with the configuration.
        model_class: Pydantic model class to validate the configuration against.
    """
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    # merge common nested tables if you use the [data]/[model]/[training] style
    if any(k in raw for k in ("data", "model", "training")):
        flat = {**raw.get("data", {}),
                **raw.get("model", {}),
                **raw.get("training", {})}
    else:
        flat = raw

    try:
        return model_class.model_validate(flat)
    except ValidationError as e:
        raise RuntimeError(f"Config validation failed for {path}\n{e}") from None
