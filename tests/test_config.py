from argparse import Namespace

import pytest
from config.myconfig import Config


def test_load_from_init_args() -> None:
    config = Config(a=1, b=2)

    # Test attribute access
    assert config.a == 1
    assert config.b == 2

    # Test that accessing a nonexistent attribute raises an error
    with pytest.raises(AttributeError):
        _ = config.c


def test_load_from_args() -> None:
    args = Namespace(a=1, b=2)
    config = Config.load_from_args(args)

    # Test attribute access
    assert config.a == 1
    assert config.b == 2

    # Test that accessing a nonexistent attribute raises an error
    with pytest.raises(AttributeError):
        _ = config.c


def test_load_from_dict() -> None:
    config_dict = {"a": 1, "b": 2}
    config = Config.load_from_dict(config_dict)

    # Test attribute access
    assert config.a == 1
    assert config.b == 2

    # Test that accessing a nonexistent attribute raises an error
    with pytest.raises(AttributeError):
        _ = config.c


def test_repr() -> None:
    config_dict = {"a": 1, "b": 2}
    config = Config.load_from_dict(config_dict)

    # Test that repr displays the config dictionary
    assert repr(config) == "Config({'a': 1, 'b': 2})"


def test_load_from_empty_args() -> None:
    args = Namespace()
    config = Config.load_from_args(args)

    # Test that config is empty and accessing any attribute raises an error
    with pytest.raises(AttributeError):
        _ = config.a


def test_load_from_empty_dict() -> None:
    config_dict = {}
    config = Config.load_from_dict(config_dict)

    # Test that config is empty and accessing any attribute raises an error
    with pytest.raises(AttributeError):
        _ = config.a


def test_config_iterator() -> None:
    config_dict = {"a": 1, "b": 2, "c": 3}
    config = Config.load_from_dict(config_dict)

    # Convert iterator results to a dict for comparison
    iterated_dict = dict(config)

    # Test that iterating over config yields the same key-value pairs as the original dict
    assert iterated_dict == config_dict

    # Test manual iteration
    items = []
    for key, value in config:
        items.append((key, value))

    # Test that all items were iterated over
    assert len(items) == len(config_dict)
    # Test that all items match
    assert dict(items) == config_dict


def test_update() -> None:
    config = Config(a=1, b=2)
    other_config = Config(b=3, c=4)
    config = config.update(other_config, d=5)
    assert config.a == 1
    assert config.b == 3
    assert config.c == 4
    assert config.d == 5
