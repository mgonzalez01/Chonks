from chonks.retrieval.query_reformulate import (
    DEFAULT_REFORMULATE_QUERY,
    MAX_EXTRACTED_TERMS,
    augment_query,
    extract_identifiers,
)


def test_default_is_off():
    assert DEFAULT_REFORMULATE_QUERY is False


# ---- extract_identifiers: extraction categories ------------------------------

def test_extracts_camel_case():
    assert extract_identifiers("call getUserById on login") == ["getUserById"]


def test_extracts_pascal_case():
    assert extract_identifiers("the UserAuth module throws") == ["UserAuth"]


def test_extracts_snake_case():
    assert extract_identifiers("check parse_one for the bug") == ["parse_one"]


def test_extracts_upper_snake_constant():
    assert extract_identifiers("raised past MAX_ITERATIONS") == ["MAX_ITERATIONS"]


def test_extracts_dotted_module_path():
    assert extract_identifiers("see foo.bar.baz for the handler") == ["foo.bar.baz"]


def test_extracts_file_path_fragment():
    assert extract_identifiers("crashes in dir/file.py on load") == ["dir/file.py"]


def test_extracts_from_backtick_span():
    assert extract_identifiers("the `parse_one` call is slow") == ["parse_one"]


def test_extracts_from_fenced_code_block():
    text = "repro:\n```\nresult = computeChecksum(buf)\n```\nthen it hangs"
    terms = extract_identifiers(text)
    assert "computeChecksum" in terms


def test_extracts_multiple_styles_in_first_seen_order():
    text = "MAX_RETRIES exceeded in getUserById, see auth/session.py, near parse_one"
    assert extract_identifiers(text) == [
        "MAX_RETRIES", "getUserById", "auth/session.py", "parse_one",
    ]


# ---- extract_identifiers: no-op / dedupe / overlap / cap ---------------------

def test_noop_on_identifier_free_prose():
    assert extract_identifiers("the response is slow and memory climbs over time") == []


def test_empty_string_returns_empty():
    assert extract_identifiers("") == []


def test_dedupes_repeated_term_keeping_first_seen_order():
    text = "parse_one fails, then parse_one is called again and parse_one throws"
    assert extract_identifiers(text) == ["parse_one"]


def test_overlapping_dotted_and_snake_prefers_longer_span():
    # "bar_baz" is a snake token wholly inside "foo.bar_baz.py"; only the
    # longer, more specific dotted span should be emitted.
    terms = extract_identifiers("open foo.bar_baz.py please")
    assert terms == ["foo.bar_baz.py"]


def test_cap_limits_extracted_term_count():
    text = " ".join(f"snake_case_{i}" for i in range(MAX_EXTRACTED_TERMS + 10))
    terms = extract_identifiers(text)
    assert len(terms) == MAX_EXTRACTED_TERMS


def test_cap_keeps_earliest_terms():
    text = " ".join(f"snake_case_{i}" for i in range(MAX_EXTRACTED_TERMS + 5))
    terms = extract_identifiers(text, max_terms=3)
    assert terms == ["snake_case_0", "snake_case_1", "snake_case_2"]


def test_single_capitalized_word_is_not_extracted():
    # No case transition, so not treated as CamelCase, same rule as
    # searcher.py's _split_identifier.
    assert extract_identifiers("Google is down") == []


# ---- augment_query: augmentation, not substitution ----------------------------

def test_disabled_is_a_pure_noop():
    query = "the getUserById call is slow"
    assert augment_query(query, enabled=False) is query


def test_enabled_appends_without_dropping_raw_text():
    query = "the getUserById call is slow"
    result = augment_query(query, enabled=True)
    assert result.startswith(query)
    assert "getUserById" in result
    assert result != query


def test_enabled_noop_on_identifier_free_prose():
    query = "the response gets slower over time"
    assert augment_query(query, enabled=True) == query


def test_enabled_dedupes_appended_tail():
    query = "parse_one fails, parse_one is called again"
    result = augment_query(query, enabled=True)
    assert result.count("parse_one") == 3  # 2 in raw text + 1 in the appended tail


# ---- augment_query: budget interplay ------------------------------------------

def test_no_budget_appends_full_tail_uncapped():
    query = "a" * 50 + " check my_distinctive_identifier"
    result = augment_query(query, enabled=True, budget_chars=None)
    assert "my_distinctive_identifier" in result


def test_within_budget_appends_full_tail():
    query = "check my_helper_fn behavior"
    result = augment_query(query, enabled=True, budget_chars=1000)
    assert result.startswith(query)
    assert "my_helper_fn" in result


def test_over_budget_truncates_raw_text_but_keeps_identifiers():
    # Raw text must be truncated, not the appended identifiers, so the terms
    # this transform exists to add are never dropped.
    long_prose = "the response gets slower and slower every single time. " * 5
    query = long_prose + "check my_distinctive_identifier next"
    budget_chars = 100
    assert len(query) > budget_chars

    result = augment_query(query, enabled=True, budget_chars=budget_chars)
    assert len(result) <= budget_chars
    assert "my_distinctive_identifier" in result


def test_over_budget_with_huge_tail_drops_raw_text_but_never_crashes():
    # Pathological: identifiers alone (post-cap) still exceed a tiny budget.
    # raw_budget floors at 0 rather than going negative; the combined result
    # can still overflow budget_chars here, that's truncate_query_text's job
    # downstream (searcher.py's _embed_query), not this function's.
    query = " ".join(f"snake_case_identifier_{i}" for i in range(MAX_EXTRACTED_TERMS))
    result = augment_query(query, enabled=True, budget_chars=20)
    assert result.startswith("\n\nIdentifiers: ")
    assert "snake_case_identifier_0" in result


def test_extract_rejects_abbreviations_and_version_numbers():
    terms = extract_identifiers(
        "The parser is slow, e.g. when parse_one runs on U.S. data "
        "with version 3.10, i.e. before SQLGlot.parse"
    )
    assert terms == ["parse_one", "SQLGlot.parse"]
