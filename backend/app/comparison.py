from difflib import SequenceMatcher
import re

from backend.app.models import (
    ApplicationData,
    ExtractedLabel,
    FieldResult,
    FieldStatus,
    VerificationResult,
    VerificationVerdict,
)


FUZZY_THRESHOLD = 90.0
ABV_TOLERANCE_PERCENTAGE_POINTS = 0.1
NET_CONTENTS_TOLERANCE_ML = 1.0

try:
    from rapidfuzz import fuzz as _rapidfuzz_fuzz
except ModuleNotFoundError:
    _rapidfuzz_fuzz = None


def _sort_tokens(value: str) -> str:
    return " ".join(sorted(value.split()))


def _fallback_token_sort_ratio(expected: str, actual: str) -> float:
    expected_sorted = _sort_tokens(expected)
    actual_sorted = _sort_tokens(actual)
    return SequenceMatcher(None, expected_sorted, actual_sorted).ratio() * 100


def _token_sort_ratio(expected: str, actual: str) -> float:
    if _rapidfuzz_fuzz is not None:
        return float(_rapidfuzz_fuzz.token_sort_ratio(expected, actual))
    return _fallback_token_sort_ratio(expected, actual)


def _stringify(value: str | float | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_text(value: str) -> str:
    no_punctuation = re.sub(r"[^\w\s]", " ", value.casefold())
    return re.sub(r"\s+", " ", no_punctuation).strip()


def _field_result(
    *,
    field: str,
    status: FieldStatus,
    expected: str | float,
    found: str | float | None,
    match_type: str,
) -> FieldResult:
    return FieldResult(
        field=field,
        match_type=match_type,
        status=status,
        expected=str(expected),
        found=_stringify(found),
    )


def _missing_actual_result(
    *,
    field: str,
    expected: str | float,
    match_type: str,
) -> FieldResult:
    return _field_result(
        field=field,
        status=FieldStatus.FAIL,
        expected=expected,
        found=None,
        match_type=match_type,
    )


def _compare_fuzzy(field: str, expected: str, actual: str | None) -> FieldResult:
    match_type = f"fuzzy_token_sort_ratio>={FUZZY_THRESHOLD:.0f}"
    if actual is None:
        return _missing_actual_result(field=field, expected=expected, match_type=match_type)

    normalized_expected = _normalize_text(expected)
    normalized_actual = _normalize_text(actual)
    score = round(_token_sort_ratio(normalized_expected, normalized_actual), 2)
    status = FieldStatus.PASS if score >= FUZZY_THRESHOLD else FieldStatus.FAIL

    return _field_result(
        field=field,
        status=status,
        expected=expected,
        found=actual,
        match_type=match_type,
    )


def compare_brand_name(expected: str, actual: str | None) -> FieldResult:
    return _compare_fuzzy("brand_name", expected, actual)


def compare_class_type(expected: str, actual: str | None) -> FieldResult:
    return _compare_fuzzy("class_type", expected, actual)


def compare_producer(expected: str, actual: str | None) -> FieldResult:
    return _compare_fuzzy("producer", expected, actual)


_COUNTRY_SYNONYMS = {
    "america": "united states",
    "usa": "united states",
    "u s a": "united states",
    "us": "united states",
    "u s": "united states",
    "united states": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "u k": "united kingdom",
    "united kingdom": "united kingdom",
    "great britain": "united kingdom",
    "england": "united kingdom",
    "scotland": "united kingdom",
    "wales": "united kingdom",
    "france": "france",
    "french republic": "france",
    "italy": "italy",
    "italia": "italy",
    "italian republic": "italy",
    "spain": "spain",
    "espana": "spain",
    "españa": "spain",
    "kingdom of spain": "spain",
    "germany": "germany",
    "deutschland": "germany",
    "federal republic of germany": "germany",
    "portugal": "portugal",
    "portuguese republic": "portugal",
    "australia": "australia",
    "commonwealth of australia": "australia",
    "argentina": "argentina",
    "argentine republic": "argentina",
    "republica argentina": "argentina",
    "republic of argentina": "argentina",
    "austria": "austria",
    "republic of austria": "austria",
    "canada": "canada",
    "chile": "chile",
    "republic of chile": "chile",
    "greece": "greece",
    "hellenic republic": "greece",
    "ireland": "ireland",
    "japan": "japan",
    "mexico": "mexico",
    "méxico": "mexico",
    "united mexican states": "mexico",
    "new zealand": "new zealand",
    "nz": "new zealand",
    "south africa": "south africa",
    "republic of south africa": "south africa",
    "netherlands": "netherlands",
    "the netherlands": "netherlands",
    "holland": "netherlands",
}

_US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC", "PR", "VI", "GU", "AS", "MP",
}
_US_STATE_NAMES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
    "maryland", "massachusetts", "michigan", "minnesota", "mississippi", "missouri",
    "montana", "nebraska", "nevada", "new hampshire", "new jersey", "new mexico",
    "new york", "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota", "tennessee",
    "texas", "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming", "district of columbia", "puerto rico",
    "u s virgin islands", "guam", "american samoa", "northern mariana islands",
}
_IMPORTER_CONTEXT_RE = re.compile(
    r"\b(?:imported\s+by|imported\s+for|importer\s+of\s+record|u\.?s\.?\s+importer)\b",
    re.IGNORECASE,
)


def _canonical_country(value: str) -> str:
    normalized = _normalize_text(value)
    normalized = re.sub(
        r"^(?:product\s+of|produced\s+in|made\s+in|country\s+of\s+origin|"
        r"imported\s+from|wine\s+of|origin)\s+",
        "",
        normalized,
    )
    return _COUNTRY_SYNONYMS.get(normalized, normalized)


def _is_us_city_state(value: str) -> bool:
    state = value.rsplit(",", maxsplit=1)
    if len(state) != 2 or not state[0].strip():
        return False
    suffix = re.sub(r"[^A-Za-z\s]", "", state[1]).strip()
    return suffix.upper() in _US_STATE_CODES or suffix.casefold() in _US_STATE_NAMES


def _has_explicit_foreign_origin(raw_text: str) -> bool:
    normalized = _normalize_text(raw_text)
    origin_prefix = r"(?:product\s+of|made\s+in|produced\s+in|imported\s+from)"
    for synonym, canonical in _COUNTRY_SYNONYMS.items():
        if canonical == "united states":
            continue
        if re.search(rf"\b{origin_prefix}\s+(?:the\s+)?{re.escape(synonym)}\b", normalized):
            return True
    return False


def _can_use_us_domestic_address(expected: str, actual: str, raw_text: str | None) -> bool:
    if _canonical_country(expected) != "united states" or not _is_us_city_state(actual):
        return False
    if raw_text is None:
        return True
    return not _IMPORTER_CONTEXT_RE.search(raw_text) and not _has_explicit_foreign_origin(raw_text)


def compare_country(
    expected: str,
    actual: str | None,
    *,
    raw_text: str | None = None,
) -> FieldResult:
    match_type = "country_synonym_exact"
    if actual is None:
        return _missing_actual_result(
            field="country_of_origin",
            expected=expected,
            match_type=match_type,
        )

    expected_country = _canonical_country(expected)
    actual_country = _canonical_country(actual)
    domestic_address_match = _can_use_us_domestic_address(expected, actual, raw_text)
    status = FieldStatus.PASS if expected_country == actual_country or domestic_address_match else FieldStatus.FAIL
    if domestic_address_match:
        match_type = "us_domestic_address"

    return _field_result(
        field="country_of_origin",
        status=status,
        expected=expected,
        found=actual,
        match_type=match_type,
    )


_PERCENT_ABV_RE = re.compile(
    r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:%|percent\b)",
    re.IGNORECASE,
)
_LABELED_ABV_AFTER_RE = re.compile(
    r"(?:abv|alc\.?\s*/?\s*vol\.?|alcohol\s+by\s+volume)\D{0,20}"
    r"(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_LABELED_ABV_BEFORE_RE = re.compile(
    r"(?<!\d)(\d+(?:\.\d+)?)\s*"
    r"(?:abv|alc\.?\s*/?\s*vol\.?|alcohol\s+by\s+volume)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)")
_PROOF_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:°\s*)?proof\b", re.IGNORECASE)


def _parse_abv(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)

    text = value.strip()
    for pattern in (_PERCENT_ABV_RE, _LABELED_ABV_AFTER_RE, _LABELED_ABV_BEFORE_RE):
        match = pattern.search(text)
        if match:
            return float(match.group(1))

    proof_match = _PROOF_RE.search(text)
    if proof_match:
        return float(proof_match.group(1)) / 2

    match = _NUMBER_RE.search(text)
    if match:
        return float(match.group(1))
    return None


def compare_abv(expected: str | float, actual: str | float | None) -> FieldResult:
    match_type = (
        "abv_numeric_normalize"
        f"+/-{ABV_TOLERANCE_PERCENTAGE_POINTS:g}_percentage_points"
    )
    expected_abv = _parse_abv(expected)
    actual_abv = _parse_abv(actual)
    if expected_abv is None or actual_abv is None:
        return _field_result(
            field="abv",
            status=FieldStatus.FAIL,
            expected=expected,
            found=actual,
            match_type=match_type,
        )

    difference = abs(expected_abv - actual_abv)
    status = (
        FieldStatus.PASS
        if difference <= ABV_TOLERANCE_PERCENTAGE_POINTS + 1e-9
        else FieldStatus.FAIL
    )

    return _field_result(
        field="abv",
        status=status,
        expected=expected,
        found=actual,
        match_type=match_type,
    )


_UNIT_TO_ML = {
    "ml": 1.0,
    "milliliter": 1.0,
    "milliliters": 1.0,
    "millilitre": 1.0,
    "millilitres": 1.0,
    "l": 1000.0,
    "liter": 1000.0,
    "liters": 1000.0,
    "litre": 1000.0,
    "litres": 1000.0,
    "cl": 10.0,
    "centiliter": 10.0,
    "centiliters": 10.0,
    "centilitre": 10.0,
    "centilitres": 10.0,
    "fl oz": 29.5735295625,
    "fluid oz": 29.5735295625,
    "fluid ounce": 29.5735295625,
    "fluid ounces": 29.5735295625,
    "floz": 29.5735295625,
    "oz": 29.5735295625,
}
_UNIT_PATTERN = "|".join(
    re.escape(unit) for unit in sorted(_UNIT_TO_ML, key=len, reverse=True)
)
_NET_CONTENTS_RE = re.compile(
    rf"(?<!\d)(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})\b",
    re.IGNORECASE,
)


def _parse_net_contents_ml(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.casefold()
    normalized = re.sub(r"\bfl\.?\s*oz\.?\b", "fl oz", normalized)
    normalized = re.sub(r"\bfluid\s+oz\.?\b", "fluid oz", normalized)
    normalized = re.sub(r"\bfloz\b", "fl oz", normalized)
    match = _NET_CONTENTS_RE.search(normalized)
    if not match:
        return None

    amount = float(match.group(1))
    unit = match.group(2).casefold()
    return amount * _UNIT_TO_ML[unit]


def compare_net_contents(expected: str, actual: str | None) -> FieldResult:
    match_type = f"net_contents_unit_normalize+/-{NET_CONTENTS_TOLERANCE_ML:g}_ml"
    expected_ml = _parse_net_contents_ml(expected)
    actual_ml = _parse_net_contents_ml(actual)
    if expected_ml is None or actual_ml is None:
        return _field_result(
            field="net_contents",
            status=FieldStatus.FAIL,
            expected=expected,
            found=actual,
            match_type=match_type,
        )

    difference = abs(expected_ml - actual_ml)
    status = (
        FieldStatus.PASS
        if difference <= NET_CONTENTS_TOLERANCE_ML + 1e-9
        else FieldStatus.FAIL
    )

    return _field_result(
        field="net_contents",
        status=status,
        expected=expected,
        found=actual,
        match_type=match_type,
    )


def compare_government_warning(
    expected: str,
    actual: str | None,
    *,
    heading_bold: bool | None = None,
) -> FieldResult:
    match_type = "exact_case_sensitive_whitespace_normalized+bold_heading"
    if actual is None:
        return _missing_actual_result(
            field="government_warning",
            expected=expected,
            match_type=match_type,
        )

    status = (
        FieldStatus.PASS
        if (
            _normalize_warning_layout(expected) == _normalize_warning_layout(actual)
            and heading_bold is True
        )
        else FieldStatus.FAIL
    )
    return _field_result(
        field="government_warning",
        status=status,
        expected=expected,
        found=actual,
        match_type=match_type,
    )


def _normalize_warning_layout(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def verify_label(
    application: ApplicationData,
    label: ExtractedLabel,
) -> VerificationResult:
    fields = [
        compare_brand_name(application.brand_name, label.brand_name),
        compare_class_type(application.class_type, label.class_type),
        compare_producer(application.producer, label.producer),
        compare_country(
            application.country_of_origin,
            label.country_of_origin,
            raw_text=label.raw_text,
        ),
        compare_abv(application.abv, label.abv),
        compare_net_contents(application.net_contents, label.net_contents),
        compare_government_warning(
            application.government_warning,
            label.government_warning,
            heading_bold=label.government_warning_heading_bold,
        ),
    ]
    verdict = (
        VerificationVerdict.APPROVED
        if all(field.status == FieldStatus.PASS for field in fields)
        else VerificationVerdict.NEEDS_REVIEW
    )
    return VerificationResult(overall_verdict=verdict, results=fields)
