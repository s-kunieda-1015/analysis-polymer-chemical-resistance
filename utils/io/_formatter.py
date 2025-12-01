from __future__ import annotations

import os
from pathlib import PurePath
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    overload,
)

if TYPE_CHECKING:
    from ._typing import PathLike

PathLikeT = TypeVar("PathLikeT", bound=os.PathLike[Any])


@overload
def format_stem(
    path: str,
    *,
    prefix: str = "",
    suffix: str = "",
    separator: str = "_",
) -> PurePath: ...


@overload
def format_stem(
    path: PathLikeT,
    *,
    prefix: str = "",
    suffix: str = "",
    separator: str = "_",
) -> PathLikeT: ...


def format_stem(
    path: PathLike,
    *,
    prefix: str | None = None,
    suffix: str | None = None,
    separator: str = "_",
) -> os.PathLike[Any]:
    purepath = path if isinstance(path, PurePath) else PurePath(path)
    stem_str = format_stem_str(purepath.stem, prefix, suffix, separator)
    return purepath.with_stem(stem_str)


def format_stem_str(
    stem: str,
    prefix: str | None = None,
    suffix: str | None = None,
    separator: str = "_",
) -> str:
    if prefix and (not prefix.endswith(separator)) and (not stem.startswith(separator)):
        prefix = prefix + separator
    else:
        prefix = ""

    if suffix and (not suffix.startswith(separator)) and (not stem.endswith(separator)):
        suffix = separator + suffix
    else:
        suffix = ""

    return prefix + stem + suffix
