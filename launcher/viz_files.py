"""Prepare visualization input."""

import shutil
import tempfile
from pathlib import Path


def stage_viz_input(source, data_dir):
    """Publish either uploaded JSON or OPFLOW output."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / "opflowout.json"

    # atomically swap out the file
    with tempfile.NamedTemporaryFile(dir=data_dir, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(source, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination
