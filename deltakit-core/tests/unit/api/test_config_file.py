# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import os
from pathlib import Path

import deltakit_core.api.environment
import deltakit_core.api.paths
import pytest


def test_read_and_write_and_override_config_file(mocker):
    mocker.patch("deltakit_core.api.constants.APP_NAME", "qec-testplorer")
    config_file = deltakit_core.api.paths.get_config_file_path()

    # reset config file
    deltakit_core.api.environment.override_persisted_variables({}, config_file)
    d = deltakit_core.api.environment.read_persisted_variables(config_file)
    assert d == {}

    # add a variable for a config file
    deltakit_core.api.environment.merge_variables({"TEST": "test123"}, config_file)
    d = deltakit_core.api.environment.read_persisted_variables(config_file)
    assert d == ({"TEST": "test123"})

    # add a new variable and override and old one
    deltakit_core.api.environment.merge_variables({"TEST": "321test", "TEST1": "4"}, config_file)
    d = deltakit_core.api.environment.read_persisted_variables(config_file)
    assert d == ({"TEST": "321test", "TEST1": "4"})

    # test that blank lines in file are ignored
    with open(config_file, "a") as f:
        f.write("\n")
    d = deltakit_core.api.environment.read_persisted_variables(config_file)
    assert d == ({"TEST": "321test", "TEST1": "4"})

    deltakit_core.api.environment.override_persisted_variables({}, config_file)
    d = deltakit_core.api.environment.read_persisted_variables(config_file)
    assert d == {}


def test_variables_reading_to_environ(mocker):
    mocker.patch("deltakit_core.api.constants.APP_NAME", "qec-testplorer")
    config_file = deltakit_core.api.paths.get_config_file_path()
    deltakit_core.api.environment.override_persisted_variables({"A": "B"}, config_file)
    deltakit_core.api.environment.load_environment_variables_from_drive()
    assert os.environ.get("A") == "B"


@pytest.mark.parametrize(
    ("platform", "path"),
    [
        ("win32", Path.home() / "AppData/Local/deltakit-explorer/.env"),
        ("darwin", Path.home() / "Library/Application Support/deltakit-explorer/.env"),
        ("linux", Path.home() / ".config/deltakit-explorer/.env")
    ]
)
def test_platform_specific_paths(mocker, platform, path):
    mocker.patch("sys.platform", platform)
    if platform == "linux" and os.getenv("XDG_CONFIG_HOME"):
        path = Path(os.getenv("XDG_CONFIG_HOME")) / "deltakit-explorer/.env"
    if platform == "win32" and os.getenv("APPDATA"):
        path = Path(os.getenv("APPDATA")) / "deltakit-explorer/.env"
    assert deltakit_core.api.paths.get_config_file_path() == path


@pytest.mark.parametrize(
    ("platform", "envvar"),
    [ ("win32", "APPDATA"), ("linux", "XDG_CONFIG_HOME")]
)
def test_platform_specific_paths_is_overridden(mocker, platform, envvar):
    os.environ[envvar] = str(Path.home() / "mock")
    mocker.patch("sys.platform", platform)
    assert deltakit_core.api.paths.get_config_file_path() == Path.home() / "mock/deltakit-explorer/.env"


def test_error_creating_the_folder(mocker):
    mocker.patch("pathlib.Path.exists", return_value=False)
    mocker.patch("pathlib.Path.mkdir", side_effect=PermissionError(""))
    assert deltakit_core.api.paths.get_config_directory() == Path.cwd()


def test_recreates_the_config_folder(mocker):
    mocker.patch("deltakit_core.api.constants.APP_NAME", "qec-testplorer")
    directory = deltakit_core.api.paths.get_config_directory()
    file = deltakit_core.api.paths.get_config_file_path()
    if file.exists():
        Path.unlink(file)
    os.removedirs(directory)
    assert not directory.exists(), "Should have been deleted"
    assert deltakit_core.api.paths.get_config_directory().exists()
    os.removedirs(directory)


@pytest.mark.parametrize("content", ["faulty_abc", "faulty_abc=abc=abc"])
def test_faulty_content(mocker, content):
    mocker.patch("deltakit_core.api.constants.APP_NAME", "qec-testplorer")
    file = deltakit_core.api.paths.get_config_file_path()
    with Path.open(file, "w", encoding="utf-8") as f:
        f.write(content)
    deltakit_core.api.environment.load_environment_variables_from_drive()
    assert os.environ.get("FAULTY_ABC") is None
