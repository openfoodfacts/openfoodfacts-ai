from dataclasses import dataclass


@dataclass
class ArgillaConfig:
    deleted_element = "#"      # Element to add in show_diff() to indicate a deleted element