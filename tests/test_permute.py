"""Behavioural tests for shader variant permutation and caching.

Every test is red on the shipped stubs: each one calls the public API and the
only failure mode is ``NotImplementedError``.  Do not edit or weaken the tests.
"""

import hashlib

import pytest

from shadervar.permute import (
    ShaderError,
    apply_hot_reload,
    cache_key,
    classify_failure,
    dedupe_variants,
    enumerate_variants,
    fallback_chain,
    normalize_key,
    select_batch,
)

BASE_SOURCE = "void main() {}\n"


def test_defect_01_feature_names_are_case_insensitive():
    upper = normalize_key({"USE_FOG": 1, "QUALITY": "high"})
    lower = normalize_key({"quality": "high", "use_fog": 1})
    assert upper == lower
    assert dict(upper) == {"quality": "high", "use_fog": 1}


def test_defect_02_boolean_spellings_share_one_key():
    truthy = [True, 1, "1", "true", "TRUE", "on", "yes"]
    normalized = {normalize_key({"USE_FOG": value}) for value in truthy}
    assert len(normalized) == 1
    assert normalized == {(("use_fog", 1),)}
    assert normalize_key({"USE_FOG": False}) == normalize_key({"USE_FOG": "0"})


def test_defect_03_variant_expansion_order_is_stable():
    keys = [{"USE_FOG": [1, 0], "QUALITY": ["low", "high"]}]
    expected = [
        (("quality", "high"), ("use_fog", 0)),
        (("quality", "high"), ("use_fog", 1)),
        (("quality", "low"), ("use_fog", 0)),
        (("quality", "low"), ("use_fog", 1)),
    ]
    assert enumerate_variants(keys, 10) == expected
    assert enumerate_variants(keys, 10) == expected


def test_defect_04_variant_expansion_is_capped():
    keys = [{f"FEATURE_{index}": [0, 1]} for index in range(20)]
    with pytest.raises(ShaderError):
        enumerate_variants(keys, 1024)


def test_defect_05_cache_key_tracks_pipeline_version():
    key = (("use_fog", 1),)
    old = cache_key(BASE_SOURCE, key, {"version": "1.0", "target": "glsl"})
    new = cache_key(BASE_SOURCE, key, {"version": "1.1", "target": "glsl"})
    assert old != new
    assert len(old) == 64


def test_defect_06_cache_key_is_insertion_order_independent():
    key = (("quality", "high"), ("use_fog", 1))
    first = cache_key(BASE_SOURCE, key, {"version": "2.0", "target": "glsl"})
    second = cache_key(BASE_SOURCE, key, {"target": "glsl", "version": "2.0"})
    assert first == second


def test_defect_07_failure_classification_is_actionable():
    assert classify_failure("ERROR: syntax error, unexpected '}'") == "syntax"
    assert classify_failure("link error: undefined reference to blur") == "link"
    assert (
        classify_failure("unsupported extension GL_EXT_foo is not supported")
        == "unsupported"
    )
    assert classify_failure("driver reset: device lost (0x887A0005)") == "driver"
    assert classify_failure("") == "driver"


def test_defect_08_fallback_chain_reaches_a_supported_variant():
    key = (("quality", 2), ("use_fog", 1))
    supported = {(("quality", 2),), ()}
    chain = fallback_chain(key, supported)
    assert chain[0] == key
    assert chain[-1] == ()
    assert len(chain) >= 2
    assert all(candidate in supported for candidate in chain[1:])


def test_defect_09_dedupe_uses_content_not_object_identity():
    first = {"key": (("use_fog", 1),), "source_hash": "a" * 64}
    duplicate = {"key": (("use_fog", 1),), "source_hash": "a" * 64}
    other = {"key": (("use_fog", 0),), "source_hash": "a" * 64}
    result = dedupe_variants([first, duplicate, other])
    assert result == [first, other]
    assert len(result) == 2


def test_defect_10_hot_reload_invalidates_only_affected_entries():
    source_hash = hashlib.sha256(BASE_SOURCE.encode("utf-8")).hexdigest()
    cache = {
        "fog": {
            "key": (("use_fog", 1),),
            "source_hash": source_hash,
            "dependencies": ("fog.glsl",),
        },
        "shadow": {
            "key": (("shadows", 1),),
            "source_hash": source_hash,
            "dependencies": ("shadow.glsl",),
        },
        "shared": {
            "key": (("quality", 1),),
            "source_hash": source_hash,
            "dependencies": ("common.glsl", "fog.glsl"),
        },
    }
    result = apply_hot_reload(BASE_SOURCE, ["fog.glsl"], cache)
    assert set(result["invalidated"]) == {"fog", "shared"}
    assert result["kept"] == 1


def test_combo_normalize_key_is_idempotent():
    once = normalize_key({"USE_FOG": True, "QUALITY": ["low", "high", "low"]})
    twice = normalize_key(dict(once))
    assert twice == once


def test_combo_empty_key_list_yields_one_empty_variant():
    assert enumerate_variants([], 1) == [()]


def test_combo_limit_equal_to_variant_count_is_allowed():
    keys = [{"USE_FOG": [0, 1]}, {"QUALITY": ["low", "high"]}]
    variants = enumerate_variants(keys, 4)
    assert len(variants) == 4
    assert len(set(variants)) == 4

    requests = [
        {"key": variant, "source_hash": "source", "cost": 1}
        for variant in variants
    ]
    batch = select_batch(requests, {}, 4)
    assert batch["compile"] == requests
    assert batch["skipped"] == []
    assert batch["used"] == 4

    extra = {"key": (("extra_feature", 1),), "source_hash": "source", "cost": 1}
    overflow = select_batch(requests + [extra], {}, 4)
    assert overflow["compile"] == requests
    assert overflow["skipped"] == [extra]


def test_combo_cache_key_same_for_different_insertion_orders():
    key = (("quality", "high"),)
    first = {"version": "3.2", "defines": {"FOG": 1, "SHADOWS": 0}}
    second = {"defines": {"SHADOWS": 0, "FOG": 1}, "version": "3.2"}
    assert cache_key(BASE_SOURCE, key, first) == cache_key(BASE_SOURCE, key, second)


def test_combo_supported_fallback_chain_has_at_least_two_entries():
    key = (("quality", 2), ("use_fog", 1))
    supported = {key, ()}
    chain = fallback_chain(key, supported)
    assert chain[0] == key
    assert chain[-1] == ()
    assert len(chain) >= 2
