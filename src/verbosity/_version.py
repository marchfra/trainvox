import importlib.metadata

try:
    if __package__ is not None:
        __version__ = importlib.metadata.version(__package__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
