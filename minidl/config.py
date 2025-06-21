try:
    import cupy  # type: ignore

    BACKEND = "cupy"
except ImportError:
    import numpy

    BACKEND = "numpy"
