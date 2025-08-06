from pathlib import Path
import tomllib
from typing import Type, TypeVar, Mapping
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

def _merge_sections(raw: Mapping, sections: list[str]) -> dict:
    """Return a flat dict: chosen tables hoisted; others left nested.
    
    Args:
        raw: Raw configuration data loaded from a TOML file.
        sections: List of section names to be flattened into the root level.
    """
    flat: dict = {}

    for key, val in raw.items():
        if key in sections and isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if sub_key in flat:
                    raise RuntimeError(
                        f"Duplicate key '{sub_key}' found while flattening '{key}'"
                    )
                flat[sub_key] = sub_val
        else:
            if key in flat:
                raise RuntimeError(f"Duplicate top-level key '{key}' in config file")
            flat[key] = val

    return flat


def load_toml_config(
    path: str | Path,
    model_class: Type[T],
    sections_to_flatten: list[str] | None = None,
) -> T:
    """
    Read TOML ➜ optionally flatten selected tables ➜ validate with Pydantic.

    Args:
        path: Path to the TOML file.
        model_class: Pydantic model class to validate the configuration.
        sections_to_flatten: List of section names to be flattened into the root level.
    """
    sections_to_flatten = sections_to_flatten or []
    path = Path(path)

    with path.open("rb") as f:
        raw = tomllib.load(f)

    flat_cfg = _merge_sections(raw, sections_to_flatten)

    try:
        return model_class.model_validate(flat_cfg)
    except ValidationError as e:
        raise RuntimeError(f"Config validation failed for {path}\n{e}") from None
