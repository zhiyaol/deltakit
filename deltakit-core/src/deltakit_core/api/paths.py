"""Defines deltakit-specific paths."""

import os
from pathlib import Path
import sys

from deltakit_core.api.constants import APP_NAME


def get_log_directory() -> Path:
    """
    Get or generate the log directory path for the user.

    Returns:
        Path: The path to the log directory based on the order of availability.
    """

    # Check for 'LOG_DIRECTORY' environment variable
    env_folder = os.getenv('LOG_DIRECTORY')
    if env_folder:
        env_data_folder = Path(env_folder)
        if env_data_folder.exists() and env_data_folder.is_dir():
            return Path(env_folder)

    return Path.cwd()


def get_config_directory() -> Path:
    """Try to obtain OS-specific configuration folder for an app.
    Fallback scenario is to use current working directory."""
    try:
        config_dir_path, override_path = None, None
        if sys.platform == "win32":
            config_dir_path = Path.home() / "AppData" / "Local" / APP_NAME
            override_path = os.getenv("APPDATA")
        elif sys.platform == "darwin":
            config_dir_path = Path.home() / "Library" / "Application Support" / APP_NAME
        else:  # Linux and other Unix-like
            config_dir_path = Path.home() / ".config" / APP_NAME
            override_path = os.getenv("XDG_CONFIG_HOME")

        if override_path is not None:
            config_dir_path = Path(override_path) / APP_NAME
        if not config_dir_path.exists():
            config_dir_path.mkdir(parents=True)
        return config_dir_path
    except PermissionError:
        return Path.cwd()


def get_config_file_path() -> Path:
    """Path to a config file. Ensure existence of the folder."""
    return get_config_directory() / ".env"
