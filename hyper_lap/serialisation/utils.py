from pathlib import Path


def to_path(path: str | Path):
    if isinstance(path, Path):
        pass
    elif isinstance(path, str):
        path = Path(path)
    else:
        raise ValueError(f"invalid path {path}")

    return path
