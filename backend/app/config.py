import os


DEFAULT_BATCH_MAX_LABELS = 10
BATCH_MAX_LABELS_ENV = "BATCH_MAX_LABELS"
DEFAULT_BATCH_UPLOAD_MAX_LABELS = 300
BATCH_UPLOAD_MAX_LABELS_ENV = "BATCH_UPLOAD_MAX_LABELS"


def batch_max_labels() -> int:
    raw = os.getenv(BATCH_MAX_LABELS_ENV, "")
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_BATCH_MAX_LABELS
    return value if 1 <= value <= 50 else DEFAULT_BATCH_MAX_LABELS


def batch_upload_max_labels() -> int:
    raw = os.getenv(BATCH_UPLOAD_MAX_LABELS_ENV, "")
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_BATCH_UPLOAD_MAX_LABELS
    return value if 1 <= value <= 300 else DEFAULT_BATCH_UPLOAD_MAX_LABELS
