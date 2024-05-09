from dataclasses import dataclass


@dataclass
class ArgillaConfig:
    deleted_element = "#"      # Element to add in show_diff() to indicate a deleted element
    html_colors = {
        "yellow": "#FCF910",
        "orange": "#EA990C",
        "red": "#E94646",
    }