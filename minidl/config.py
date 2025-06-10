try:
    import cupy as np  # type: ignore

    BACKEND = "cupy"
except ImportError:
    import numpy as np

    BACKEND = "numpy"
