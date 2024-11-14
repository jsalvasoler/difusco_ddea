from argparse import Namespace

import pytest
from ea.config import Config


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

def test_prevent_new_attribute_assignment() -> None:
    config_dict = {"a": 1, "b": 2}
    config = Config.load_from_dict(config_dict)

    # Test that setting a new attribute raises an error
    with pytest.raises(AttributeError):
        config.c = 3

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

def test_no_direct_modification_of_config() -> None:
    config_dict = {"a": 1, "b": 2}
    config = Config.load_from_dict(config_dict)

    # Test that modifying an existing attribute directly raises an error
    with pytest.raises(AttributeError):
        config.a = 10

def test_only_internal_config_can_be_set() -> None:
    config = Config()
    # Test that _config can be set directly without error
    config._config = {"test": 42}   # noqa: SLF001
    assert config.test == 42
