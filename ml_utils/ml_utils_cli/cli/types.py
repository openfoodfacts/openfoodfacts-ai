import enum


class ExportSource(enum.StrEnum):
    hf = "hf"
    ls = "ls"


class ExportDestination(enum.StrEnum):
    hf = "hf"
    ultralytics = "ultralytics"


class TaskType(enum.StrEnum):
    object_detection = "object_detection"
    classification = "classification"
