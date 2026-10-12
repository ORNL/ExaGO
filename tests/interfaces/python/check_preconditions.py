import os
from pathlib import Path


def check_import():
    """Ensure the bindings from the current build can be imported."""
    try:
        import exago  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Could not import the ExaGO bindings."
        ) from error


def datafile(*parts):
    """Find test input data."""
    root = Path(os.environ.get(
        "EXAGO_TEST_DATA_DIR", Path(__file__).resolve().parents[3] / "datafiles"
    ))
    return str(root.joinpath(*parts))


def check_preconditions():
    """Ensure all preconditions for testing Python bindings have been met."""
    check_import()
