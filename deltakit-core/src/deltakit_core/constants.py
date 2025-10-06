from enum import Enum

class RiverlaneColors(Enum):
    """Riverlane colors for visualisation."""

    LIGHT_GREEN = "#00968f"
    RUST = "#dc4405"
    BLUE = "#3ccbda"
    DARK_GREEN = "#1d3c34"
    BLUE_70_PERC = "#77dbe5"
    ORANGE = "#ff7500"
    MILD_GREEN = "#006f62"
    ORANGE_70_PERC = "#ff9e4d"
    BLACK_70_PERC = "#777776"


RIVERLANE_COLORS_LIST = [color.value for color in RiverlaneColors]
