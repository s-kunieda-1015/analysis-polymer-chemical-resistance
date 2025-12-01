from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
    Union,
)

if TYPE_CHECKING:
    import os

PathLike: TypeAlias = Union[str, "os.PathLike[Any]"]
