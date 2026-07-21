from backend.app.comparison import (
    compare_abv,
    compare_brand_name,
    compare_country,
    compare_government_warning,
    compare_net_contents,
    compare_class_type,
    compare_producer,
    verify_label,
)
from backend.app.models import (
    ApplicationData,
    ExtractedLabel,
    FieldStatus,
    VerificationVerdict,
)


WARNING = "GOVERNMENT WARNING: THIS IS THE EXACT WARNING TEXT"


def passing_application() -> ApplicationData:
    return ApplicationData(
        brand_name="Sunset Ridge",
        class_type="Cabernet Sauvignon",
        producer="North Valley Estate Winery LLC",
        country_of_origin="United States",
        abv="45%",
        net_contents="750 mL",
        government_warning=WARNING,
    )


def passing_label() -> ExtractedLabel:
    return ExtractedLabel(
        brand_name="SUNSET RIDGE",
        class_type="Sauvignon Cabernet",
        producer="North Valley Estate Winery, LLC",
        country_of_origin="USA",
        abv="45% Alc./Vol. (90 Proof)",
        net_contents="750ml",
        government_warning=WARNING,
        government_warning_heading_bold=True,
    )


def assert_pass(result) -> None:
    assert result.status == FieldStatus.PASS


def assert_fail(result) -> None:
    assert result.status == FieldStatus.FAIL


def test_brand_case_only_difference_passes() -> None:
    result = compare_brand_name("Sunset Ridge", "SUNSET RIDGE")

    assert_pass(result)
    assert result.found == "SUNSET RIDGE"
    assert result.match_type == "fuzzy_token_sort_ratio>=90"


def test_fuzzy_fields_handle_spacing_punctuation_and_word_order() -> None:
    class_type = compare_class_type("Cabernet Sauvignon", "Sauvignon Cabernet")
    producer = compare_producer(
        "North Valley Estate Winery LLC",
        " North Valley Estate Winery, LLC ",
    )

    assert_pass(class_type)
    assert class_type.found == "Sauvignon Cabernet"
    assert_pass(producer)
    assert producer.found == " North Valley Estate Winery, LLC "


def test_fuzzy_field_clearly_different_fails() -> None:
    result = compare_brand_name("Sunset Ridge", "Harbor Point")

    assert_fail(result)
    assert result.found == "Harbor Point"


def test_country_usa_matches_united_states() -> None:
    result = compare_country("United States", "USA")

    assert_pass(result)


def test_country_synonym_with_punctuation_matches() -> None:
    result = compare_country("United States of America", "U.S.A.")

    assert_pass(result)


def test_country_product_of_prefix_matches_country() -> None:
    result = compare_country("USA", "Product of USA")

    assert_pass(result)


def test_country_common_non_us_synonyms_match() -> None:
    assert_pass(compare_country("France", "Product of French Republic"))
    assert_pass(compare_country("Italy", "Italia"))
    assert_pass(compare_country("Spain", "España"))
    assert_pass(compare_country("Germany", "Deutschland"))
    assert_pass(compare_country("Portugal", "Portuguese Republic"))
    assert_pass(compare_country("Australia", "Commonwealth of Australia"))


def test_country_different_country_fails() -> None:
    result = compare_country("United States", "Canada")

    assert_fail(result)
    assert result.found == "Canada"


def test_us_city_state_matches_expected_usa() -> None:
    result = compare_country("USA", "KINGSTON, NY")

    assert_pass(result)
    assert result.found == "KINGSTON, NY"
    assert result.match_type == "us_domestic_address"


def test_us_city_full_state_name_matches_expected_usa() -> None:
    assert_pass(compare_country("United States", "Portland, Oregon"))


def test_us_importer_address_does_not_imply_us_origin() -> None:
    result = compare_country(
        "USA",
        "Miami, FL",
        raw_text="PRODUCT OF CANADA. IMPORTED BY SAMPLE IMPORTS, MIAMI, FL.",
    )

    assert_fail(result)
    assert result.match_type == "country_synonym_exact"


def test_explicit_foreign_origin_blocks_us_address_fallback() -> None:
    result = compare_country(
        "USA",
        "Miami, FL",
        raw_text="PRODUCT OF CANADA. BOTTLED FOR SAMPLE COMPANY, MIAMI, FL.",
    )

    assert_fail(result)


def test_us_address_does_not_match_non_us_expected_country() -> None:
    assert_fail(compare_country("Canada", "KINGSTON, NY"))


def test_verify_label_uses_raw_text_for_domestic_address_context() -> None:
    label = passing_label().model_copy(
        update={
            "country_of_origin": "KINGSTON, NY",
            "raw_text": "DISTILLED AND BOTTLED BY SAMPLE DISTILLERY, KINGSTON, NY.",
        }
    )

    result = verify_label(passing_application(), label)

    assert result.overall_verdict == VerificationVerdict.APPROVED


def test_abv_ignores_proof_when_percent_is_present() -> None:
    result = compare_abv("45%", "45% Alc./Vol. (90 Proof)")

    assert_pass(result)


def test_abv_normalizes_percent_text_and_numeric_values() -> None:
    result = compare_abv(13.5, "Alc. 13.5% by Vol.")

    assert_pass(result)


def test_abv_value_inside_tolerance_passes() -> None:
    result = compare_abv("13.5%", "13.6%")

    assert_pass(result)


def test_abv_value_outside_tolerance_fails() -> None:
    result = compare_abv("13.5%", "14.0%")

    assert_fail(result)
    assert result.found == "14.0%"


def test_abv_proof_without_percent_converts_to_abv() -> None:
    result = compare_abv("45%", "90 Proof")

    assert_pass(result)


def test_abv_unparseable_value_fails() -> None:
    result = compare_abv("13.5%", "unknown")

    assert_fail(result)
    assert result.found == "unknown"


def test_net_contents_allows_no_space_between_amount_and_unit() -> None:
    result = compare_net_contents("750 mL", "750ml")

    assert_pass(result)


def test_net_contents_normalizes_liters_and_centiliters() -> None:
    liters = compare_net_contents("750 mL", "0.75 L")
    centiliters = compare_net_contents("750 mL", "75 cl")

    assert_pass(liters)
    assert_pass(centiliters)


def test_net_contents_normalizes_us_fluid_ounces() -> None:
    assert_pass(compare_net_contents("355 mL", "12 FL OZ"))
    assert_pass(compare_net_contents("355 mL", "12 oz"))


def test_net_contents_different_volume_fails() -> None:
    result = compare_net_contents("750 mL", "700 mL")

    assert_fail(result)
    assert result.found == "700 mL"


def test_net_contents_unparseable_value_fails() -> None:
    result = compare_net_contents("750 mL", "one bottle")

    assert_fail(result)
    assert result.found == "one bottle"


def test_government_warning_title_case_fails() -> None:
    result = compare_government_warning(
        WARNING,
        "Government Warning: This Is The Exact Warning Text",
    )

    assert_fail(result)


def test_government_warning_missing_colon_fails() -> None:
    result = compare_government_warning(
        WARNING,
        "GOVERNMENT WARNING THIS IS THE EXACT WARNING TEXT",
    )

    assert_fail(result)


def test_government_warning_missing_value_fails() -> None:
    result = compare_government_warning(WARNING, None)

    assert_fail(result)
    assert result.found is None


def test_government_warning_exact_all_caps_passes() -> None:
    result = compare_government_warning(WARNING, WARNING, heading_bold=True)

    assert_pass(result)


def test_government_warning_non_bold_heading_fails() -> None:
    result = compare_government_warning(WARNING, WARNING, heading_bold=False)

    assert_fail(result)


def test_government_warning_unknown_heading_weight_fails() -> None:
    result = compare_government_warning(WARNING, WARNING, heading_bold=None)

    assert_fail(result)


def test_government_warning_line_breaks_and_indentation_pass() -> None:
    actual = "  GOVERNMENT WARNING:\n    THIS IS THE EXACT\n  WARNING TEXT  "

    result = compare_government_warning(WARNING, actual, heading_bold=True)

    assert_pass(result)
    assert result.found == actual


def test_government_warning_failure_preserves_extracted_text() -> None:
    misread_warning = "GOVERNMENT WARNlNG: THIS IS THE EXACT WARNING TEXT"

    result = compare_government_warning(WARNING, misread_warning)

    assert_fail(result)
    assert result.found == misread_warning


def test_verify_label_all_fields_pass() -> None:
    result = verify_label(passing_application(), passing_label())

    assert result.overall_verdict == VerificationVerdict.APPROVED
    assert len(result.results) == 7
    assert all(field.status == FieldStatus.PASS for field in result.results)


def test_verify_label_any_failure_needs_review() -> None:
    label = passing_label().model_copy(update={"government_warning": WARNING.lower()})

    result = verify_label(passing_application(), label)

    assert result.overall_verdict == VerificationVerdict.NEEDS_REVIEW
    assert any(
        field.field == "government_warning" and field.status == FieldStatus.FAIL
        for field in result.results
    )
