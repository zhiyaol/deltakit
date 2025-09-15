# (c) Copyright Riverlane 2020-2025.
import importlib.metadata

__version__ = importlib.metadata.version(__package__)

# Prevent import of `importlib` (and any other non-public objects) from this module.
del importlib
