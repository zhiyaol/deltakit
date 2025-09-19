"""Defines constants for the cloud API."""
from typing import Final


APP_NAME: Final[str] = "deltakit-explorer"
DELTAKIT_SERVER_URL_ENV: Final[str] = "DELTAKIT_SERVER"
DELTAKIT_SERVER_DEFAULT_URL_ENV: Final[str] = "https://deltakit.riverlane.com/proxy"
HTTP_PACKET_LIMIT: Final[int] = 20_000_000
