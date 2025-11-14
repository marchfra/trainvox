import importlib.metadata

try:
    if __package__ is not None:
        __version__ = importlib.metadata.version(__package__)
    # __version__ = importlib.metadata.version("verbosity-strategy")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
