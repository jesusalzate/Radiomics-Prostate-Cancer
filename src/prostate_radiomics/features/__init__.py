"""Feature table construction and selection.

Submodules are intentionally not imported eagerly so lightweight feature-table
commands do not require the full ML stack.
"""

__all__ = ["concatenate", "selection"]
