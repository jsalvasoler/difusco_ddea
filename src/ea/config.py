# ruff: noqa: ANN102

from argparse import Namespace
from typing import Iterator


class Config:
    def __init__(self, **kwargs) -> None:
        self._config = kwargs

    @classmethod
    def load_from_args(cls, args: Namespace) -> "Config":
        config = cls()
        config._config = vars(args)  # noqa: SLF001
        return config

    @classmethod
    def load_from_dict(cls, config_dict: dict) -> "Config":
        config = cls()
        config._config = config_dict  # noqa: SLF001
        return config

    def __getattr__(self, name: str) -> any:
        if name in self._config:
            return self._config[name]
        error_msg = f"'Config' object has no attribute '{name}'"
        raise AttributeError(error_msg)

    def __setattr__(self, name: str, value: any) -> None:
        if name == "_config":  # Allow setting _config attribute
            super().__setattr__(name, value)
        else:
            error_msg = f"'Config' object has no attribute '{name}'"
            raise AttributeError(error_msg)

    def __iter__(self) -> Iterator[tuple[str, any]]:
        """Iterate over config key-value pairs."""
        return iter(self._config.items())

    def __repr__(self) -> str:
        return f"Config({self._config})"
