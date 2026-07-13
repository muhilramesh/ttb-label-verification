class VisionServiceError(Exception):
    """Base error for image extraction failures."""


class VisionConfigurationError(VisionServiceError):
    pass


class VisionInvalidImageError(VisionServiceError):
    pass


class VisionTimeoutError(VisionServiceError):
    pass


class VisionRateLimitError(VisionServiceError):
    pass


class VisionProviderError(VisionServiceError):
    pass


class VisionParseError(VisionServiceError):
    pass
