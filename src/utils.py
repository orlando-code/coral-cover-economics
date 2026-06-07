import re

import numpy as np


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use in filenames."""
    # Replace problematic characters with underscores
    name = re.sub(r"[%/\\:*?\"<>|() ]", "_", name)
    # Replace multiple underscores with single
    name = re.sub(r"_+", "_", name)
    # Remove leading/trailing underscores
    name = name.strip("_")
    return name.lower()


def make_json_safe(value: float) -> float:
    """Make a value JSON safe."""
    if np.isnan(value):
        return None
    return float(value)
