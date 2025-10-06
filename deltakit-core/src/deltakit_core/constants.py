from enum import Enum


class RiverlaneColors(Enum):
    """Riverlane colors for visualisation."""

    LIGHT_GREEN = "#00968f"
    RUST = "#dc4405"
    DARK_GREEN = "#1d3c34"
    BLUE_70_PERC = "#77dbe5"
    ORANGE = "#ff7500"
    MILD_GREEN = "#006f62"
    BLUE = "#3ccbda"
    ORANGE_70_PERC = "#ff9e4d"
    BLACK_70_PERC = "#777776"

RIVERLANE_COLORS_LIST = [color.value for color in RiverlaneColors]

_RIVERLANE_PLOT_COLOURS: list[str] = [
    "#006F62",
    "#FF6A00",
    "#4B5BFF",
    "#E01500",
    "#003B40",
    "#D835A2",
    "#00A3A3",
    "#9B3EC6",
    "#54170E",
    "#FFD00B",
    "#0AD2F2",
]
_RIVERLANE_DARK_TEXT_COLOUR = "#0A1600"
_RIVERLANE_WHITE_TEXT_COLOUR = "#E5F0EF"
_RIVERLANE_LINE_COLOUR = "#003B40"
_RIVERLANE_NEUTRAL_GREY = "#DBDBDB"
