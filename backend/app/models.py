from enum import Enum

from pydantic import BaseModel, ConfigDict


class FieldStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class VerificationVerdict(str, Enum):
    APPROVED = "APPROVED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class BatchItemStatus(str, Enum):
    APPROVED = "APPROVED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    ERROR = "ERROR"


class ApplicationData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brand_name: str
    class_type: str
    producer: str
    country_of_origin: str
    abv: str | float
    net_contents: str
    government_warning: str


class ExtractedLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brand_name: str | None = None
    class_type: str | None = None
    producer: str | None = None
    country_of_origin: str | None = None
    abv: str | float | None = None
    net_contents: str | None = None
    government_warning: str | None = None
    government_warning_heading_bold: bool | None = None
    raw_text: str | None = None
    extraction_confidence: float | None = None


class FieldResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    match_type: str
    expected: str
    found: str | None
    status: FieldStatus


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_verdict: VerificationVerdict
    results: list[FieldResult]
    extracted_label: ExtractedLabel | None = None
    latency_ms: int | None = None


class BatchItemError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class BatchItemResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    filename: str | None = None
    status: BatchItemStatus
    result: VerificationResult | None = None
    error: BatchItemError | None = None
    latency_ms: int


class BatchSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: int
    needs_review: int
    total: int
    errors: int = 0
    latency_ms: int | None = None


class BatchVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[BatchItemResult]
    summary: BatchSummary
