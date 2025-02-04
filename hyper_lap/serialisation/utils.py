from pathlib import Path


def as_path(path: str | Path):
    if isinstance(path, Path):
        pass
    elif isinstance(path, str):
        path = Path(path)
    else:
        raise ValueError(f"invalid path {path}")

    return path
