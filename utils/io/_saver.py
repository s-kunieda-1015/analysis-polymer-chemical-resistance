from __future__ import annotations

import inspect
import json
import logging
import pickle
import warnings

# import tomli_w
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    Generic,
    ParamSpec,
    Protocol,
    TypeVar,
    overload,
)

import joblib  # type: ignore[import]
from pandas.core.generic import NDFrame

from ._formatter import format_stem

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
    )
    from logging import Logger
    from types import FrameType

    import pandas as pd
    from matplotlib.figure import Figure  # type: ignore[import]

    from ._typing import PathLike


V_contra = TypeVar("V_contra", contravariant=True)
P = ParamSpec("P")

_root_logger = logging.getLogger()


def _concat_with_or(*args: str) -> str:
    if len(args) == 1:
        return args[0]
    return ", ".join(args[:-1]) + f" or {args[-1]}"


class InconsistentFlagsError(ValueError):
    """Exception raised when flags are inconsistent."""

    def __init__(self, one: str, others: str | Iterable[str]) -> None:
        if not isinstance(others, str):
            others = _concat_with_or(*others)
        super().__init__(f"{one} cannot be specified with {others}.")


class CannotDetectVariableNameError(ValueError):
    """Exception raised when variable name cannot be detected."""

    def __init__(self) -> None:
        super().__init__("Cannot detect variable name.")


def get_varname(var: Any, frame: FrameType) -> str:
    """Attempt to detect variable name.

    Args:
        var (Any): Variable
        frame (FrameType): Frame where the variable is defined

    Raises:
        CannotDetectVariableNameError: If variable name detection fails

    Returns:
        str: Variable name
    """
    for k, v in frame.f_locals.items():
        if v is var:
            return k
    raise CannotDetectVariableNameError


class Saver(Protocol, Generic[V_contra, P]):
    @overload
    def __call__(  # noqa: PLR0913
        self,
        value: V_contra,
        fullpath: PathLike,
        directory: None = None,
        stem: None = None,
        logger: Logger | None = _root_logger,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Path: ...

    @overload
    def __call__(  # noqa: PLR0913
        self,
        value: V_contra,
        fullpath: None = None,
        directory: PathLike | None = ...,
        stem: PathLike | None = ...,
        logger: Logger | None = _root_logger,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Path: ...

    def __call__(  # noqa: PLR0913
        self,
        value: V_contra,
        fullpath: PathLike | None = None,
        directory: PathLike | None = None,
        stem: PathLike | None = None,
        logger: Logger | None = _root_logger,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Path: ...


def saver(file_suffix: str):  # noqa: ANN202  # type: ignore[no-untyped-def]
    def wrapper(
        func: Callable[Concatenate[V_contra, Path, P], None],
    ) -> Saver[V_contra, P]:
        @wraps(func)
        def _wrapper(
            value: V_contra,
            fullpath: PathLike | None = None,
            directory: PathLike | None = None,
            stem: PathLike | None = None,
            logger: Logger | None = _root_logger,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Path:
            if fullpath:
                if directory or stem:
                    raise InconsistentFlagsError(
                        "fullpath",  # noqa: EM101
                        ("directory", "stem"),
                    )
                filepath = Path(fullpath)
            else:
                if not stem:
                    stem = get_varname(value, inspect.stack()[1].frame)
                filepath = Path(stem) if directory is None else Path(directory, stem)

            filepath = filepath.resolve().with_suffix(file_suffix)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            func(value, filepath, *args, **kwargs)
            if logger:
                logger.info(f"Saved to '{filepath}'.")

            return filepath

        return _wrapper

    return wrapper


def _save_as_csv(
    value: pd.DataFrame | pd.Series[Any],
    path: Path,
    default_index_name: str | None = "index",
    **kwargs: Any,
) -> None:
    if default_index_name is not None and (not value.index.name):  # pyright: ignore
        value.index.name = default_index_name
    value.to_csv(path, **kwargs)


@saver(".csv")
def save_as_csv(
    value: pd.DataFrame | pd.Series[Any],
    path: Path,
    default_index_name: str | None = "index",
    **kwargs: Any,
) -> None:
    _save_as_csv(value, path, default_index_name, **kwargs)


@saver(".csv")
def save_as_csv_with_head(
    value: pd.DataFrame | pd.Series[Any],
    path: Path,
    n_heads: int = 5,
    **kwargs: Any,
) -> None:
    save_as_csv(value.head(n_heads), format_stem(path, suffix="head"), **kwargs)
    _save_as_csv(value, path, **kwargs)


@saver(".json")
def save_as_json(value: Any, path: Path, **kwargs: Any) -> None:
    with path.open("w") as f:
        json.dump(value, f, ensure_ascii=False, **kwargs)


@saver(".pickle")
def save_as_pickle(value: Any, path: Path, **kwargs: Any) -> None:
    if isinstance(value, NDFrame):
        value.to_pickle(path, **kwargs)
    else:
        with path.open("wb") as f:
            pickle.dump(value, f, **kwargs)


@saver(".joblib")
def save_as_joblib(value: Any, path: Path, compress: int = 3, **kwargs: Any) -> None:
    joblib.dump(value, path, compress, **kwargs)  # type: ignore


@saver(".png")
def save_as_png(
    figure: Figure,
    path: Path,
    *,
    sets_tight_layout: bool = True,
    **kwargs: Any,
) -> None:
    if sets_tight_layout:
        # Ignore UserWarning to work around the following bug in Python 3.11
        # https://github.com/matplotlib/matplotlib/issues/26290
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            figure.tight_layout()
    figure.savefig(path, **kwargs)  # type: ignore


# @saver(".toml")
# def save_as_toml(value: dict[str, Any], path: Path, **kwargs: Any) -> None:
#     with open(path, "wb") as f:
#         tomli_w.dump(value, f, **kwargs)
