class ObjectDetectionError(Exception):
    """Base class for user-facing application errors."""


class ConfigurationError(ObjectDetectionError):
    """Raised for an invalid command or configuration."""


class SourceError(ObjectDetectionError):
    """Raised when an image, video, or camera cannot be opened."""


class ModelError(ObjectDetectionError):
    """Raised when a model cannot be resolved or loaded."""


class DatasetError(ObjectDetectionError):
    """Raised when dataset integrity checks fail."""

