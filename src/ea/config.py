# ruff: noqa: ANN102

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from argparse import Namespace


class Config:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def load_from_args(cls, args: Namespace) -> Config:
        return cls(**vars(args))

    @classmethod
    def load_from_dict(cls, config_dict: dict) -> Config:
        return cls(**config_dict)

    def __getattr__(self, name: str) -> any:
        error_msg = f"'Config' object has no attribute '{name}'"
        raise AttributeError(error_msg)

    def __iter__(self) -> Iterator[tuple[str, any]]:
        return iter(self.__dict__.items())

    def __repr__(self) -> str:
        return f"Config({self.__dict__})"
