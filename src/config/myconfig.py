# ruff: noqa: ANN102

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from argparse import Namespace

from ast import literal_eval


class Config:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

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

    def update(self, other_config: Config, **kwargs) -> Config:
        new_config = Config(**self.__dict__)
        for key, value in other_config:
            setattr(new_config, key, value)
        for key, value in kwargs.items():
            setattr(new_config, key, value)
        return new_config

    @staticmethod
    def load_saved_config(config_name: str) -> Config:
        if config_name.endswith(".py"):
            config_name = config_name[:-3]

        match config_name:
            case "mis_inference":
                from config.configs.mis_inference import config
            case "tsp_inference":
                from config.configs.tsp_inference import config
            case _:
                raise ValueError(f"Invalid config name: {config_name}")

        return config

    @staticmethod
    def load_from_namespace(args: Namespace) -> Config:
        return Config(**vars(args))

    @staticmethod
    def load_from_args(args: Namespace, extra: list[str]) -> Config:
        # 1. Lowest priority: saved config
        config = Config.load_saved_config(args.config_name)

        # 2. Middle priority: args from the arg parser, can have default values overriding saved config
        args_config = Config.load_from_namespace(args)
        config = config.update(args_config)

        # 3. Highest priority: extra args from the command line
        for i in range(len(extra) - 1):
            current, next_elem = extra[i], extra[i + 1]
            if current.startswith("--") and not next_elem.startswith("--"):
                key = current[2:]
                try:
                    value = literal_eval(next_elem)
                except (SyntaxError, ValueError):
                    value = next_elem
                setattr(config, key, value)

        return config
