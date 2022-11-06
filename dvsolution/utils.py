import h5py
import click
import numpy as np
from pathlib import Path
from typing import List


def validate_inputs(path: Path, files: List[Path]):

    for file in files:
        if not (path / Path(file)).exists():
            raise click.FileError(file, f"need files {files}")
