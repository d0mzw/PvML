import torch as t


def get_device() -> t.device:
    """Scripts call this. Modules take their device from their parameters."""
    if t.cuda.is_available():  # ROCm reports as cuda
        return t.device("cuda")
    if t.backends.mps.is_available():
        return t.device("mps")
    return t.device("cpu")
