import click
from typing import List
from pathlib import Path


def validate_inputs(path: Path, files: List[Path]):
    """ validates existence of files """
    for file in files:
        if not (path / Path(file)).exists():
            raise click.FileError(file, f"need files {files}")
