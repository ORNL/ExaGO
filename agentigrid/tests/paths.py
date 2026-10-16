"""Test resolution of ExaGO binaries."""

import os
import shutil
from pathlib import Path


def get_binary_dir() -> Path:
    if os.environ.get("EXAGO_TEST_BINARY_DIR"):
        return Path(os.environ["EXAGO_TEST_BINARY_DIR"])
    if os.environ.get("AGENTIGRID_CONFIG_DIR"):
        from agentigrid.config import get_default_config_path, load_config
        return load_config(get_default_config_path()).exago.binary_dir
    project_root = Path(__file__).resolve().parent.parent
    for candidate in (
        project_root.parent / "build/applications",
        project_root.parent / "build/bin",
        project_root / "applications/exago",
    ):
        if (candidate / "opflow").is_file():
            return candidate
    binary = shutil.which("opflow")
    return Path(binary).parent if binary else project_root / "applications/exago"
