"""Shader variant permutation, cache keys and hot-reload bookkeeping.

A *variant* is represented inline as a small dict::

    {"key": (("quality", 1), ("use_fog", 0)), "source_hash": "..."}

``key`` is the canonical tuple returned by :func:`normalize_key`: feature
names are lower-case, boolean-ish values are ``0`` / ``1``, and feature pairs
are sorted.  The helpers in this module are deterministic, use only the Python
standard library, and never read the clock, the network or random state.

The hot-reload cache used by :func:`apply_hot_reload` maps an opaque cache key
to metadata::

    {
        "<cache key>": {
            "key": tuple,                  # normalized variant key
            "source_hash": str,            # sha256(base_source) when cached
            "dependencies": ("fog.glsl",), # files that feed this variant
        },
    }

The empty key ``()`` is the all-defaults fallback variant.
"""

import hashlib
import json
from collections.abc import Mapping
from itertools import product

__all__ = [
    "ShaderError",
    "normalize_key",
    "enumerate_variants",
    "cache_key",
    "classify_failure",
    "fallback_chain",
    "dedupe_variants",
    "select_batch",
    "apply_hot_reload",
]

_TRUTHY_STRINGS = frozenset({"1", "true", "on", "yes"})
_FALSY_STRINGS = frozenset({"0", "false", "off", "no"})

_SYNTAX_MARKERS = ("syntax", "parse error", "unexpected token", "expected")
_LINK_MARKERS = ("link", "undefined reference", "unresolved external", "duplicate symbol")
_UNSUPPORTED_MARKERS = (
    "unsupported",
    "not supported",
    "requires feature",
    "extension unavailable",
)


class ShaderError(Exception):
    """Raised for invalid shader-variant input or an impossible request."""


def _canonical_value(value):
    """Canonicalize one feature value (booleans, strings, lists)."""
    # bool is a subclass of int, so it must be checked before plain scalars.
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in _TRUTHY_STRINGS:
            return 1
        if lowered in _FALSY_STRINGS:
            return 0
        return value
    if isinstance(value, list):
        items = [_canonical_value(item) for item in value]
        try:
            unique = list(frozenset(items))
        except TypeError as exc:
            raise ShaderError("a list feature contains an unhashable item") from exc
        try:
            unique.sort()
        except TypeError as exc:
            raise ShaderError("a list feature contains items that cannot be ordered") from exc
        return tuple(unique)
    # Tuples are already canonical and are preserved; other scalars pass through.
    return value


def normalize_key(key: dict) -> tuple:
    """Return the canonical, hashable form of one shader feature mapping.

    Canonicalization rules:

    * feature names are lower-cased and sorted ascending;
    * booleans become ``1`` / ``0``; the strings ``"1"``, ``"true"``,
      ``"on"`` and ``"yes"`` (case-insensitive) become ``1``, while ``"0"``,
      ``"false"``, ``"off"`` and ``"no"`` become ``0``;
    * list values are canonicalized item by item, de-duplicated and sorted, and
      returned as tuples; tuples are already canonical and are preserved;
    * all other scalar values are preserved.

    Args:
        key: mapping with string feature names.  A list value means a set-like
            multivalue feature, not a set of alternatives (use
            :func:`enumerate_variants` to expand alternatives).

    Returns:
        A tuple of ``(name, value)`` pairs sorted by name, so equal mappings
        built in different insertion orders compare equal.  Feeding
        ``dict(result)`` back into this function returns an equal tuple
        (idempotent).

    Raises:
        ShaderError: if ``key`` is not a mapping, a feature name is not a
            string, or a value cannot be made hashable.
    """
    if not isinstance(key, Mapping):
        raise ShaderError("variant key must be a mapping of feature names")
    pairs = []
    for name, value in key.items():
        if not isinstance(name, str):
            raise ShaderError("feature names must be strings")
        pairs.append((name.lower(), _canonical_value(value)))
    try:
        pairs.sort()
        result = tuple(pairs)
        hash(result)
    except TypeError as exc:
        raise ShaderError("variant key contains a value that cannot be made hashable") from exc
    return result


def enumerate_variants(keys: list[dict], max_variants: int) -> list[tuple]:
    """Expand feature alternatives into a deterministic Cartesian product.

    Each dict in ``keys`` maps feature names to either one scalar value or a
    list of alternatives.  Feature names are lower-cased; a list is expanded as
    alternatives after canonicalization, de-duplication and sorting.  A dict
    may contain several features and therefore contribute several product
    axes.

    Ordering is stable and does not depend on set iteration:

    * input dicts are visited in list order;
    * inside one dict, feature names are sorted ascending;
    * each feature's alternatives are sorted ascending;
    * the left-most axis is the outermost loop.

    The output is the list of normalized key tuples.  ``keys=[]`` produces the
    single empty variant ``[()]``.  The product size is checked before the
    product is materialized, so an oversized request fails without allocating
    the full result.

    Args:
        keys: list of feature dictionaries; each may contain multiple features.
        max_variants: maximum number of variants to return, a non-negative
            integer.  A value exactly equal to the product size is allowed.

    Returns:
        A list of variant key tuples in product order; its length never exceeds
        ``max_variants``.

    Raises:
        ShaderError: ``max_variants`` is negative, the input is malformed, or
            the Cartesian product would contain more than ``max_variants``
            variants.  The size guard runs before generating the full product.
    """
    if not isinstance(keys, list):
        raise ShaderError("keys must be a list of feature dictionaries")
    if not isinstance(max_variants, int) or max_variants < 0:
        raise ShaderError("max_variants must be a non-negative integer")

    axes = []  # (feature name, tuple of alternatives), in expansion order
    for feature_dict in keys:
        if not isinstance(feature_dict, Mapping):
            raise ShaderError("each entry of keys must be a feature dictionary")
        for name, value in normalize_key(feature_dict):
            alternatives = value if isinstance(value, tuple) else (value,)
            axes.append((name, alternatives))

    # Size guard first: never materialize an oversized Cartesian product.
    total = 1
    for _, alternatives in axes:
        total *= len(alternatives)
    if total > max_variants:
        raise ShaderError(
            f"variant expansion would yield {total} variants, "
            f"exceeding the cap of {max_variants}"
        )

    if not axes:
        return [()]

    names = [name for name, _ in axes]
    variants = []
    for combination in product(*(alternatives for _, alternatives in axes)):
        # Re-sort the pairs so every emitted variant is a normalized key.
        variants.append(tuple(sorted(zip(names, combination))))
    return variants


def cache_key(base_source: str, key: tuple, pipeline: dict) -> str:
    """Return a stable SHA-256 digest for one compiled variant.

    The digest covers every input that can change compiled output:

    * ``sha256(base_source.encode("utf-8"))``;
    * the normalized key (feature pairs are sorted, so key order is
      irrelevant);
    * the complete ``pipeline`` mapping, serialized as canonical JSON with
      sorted keys at every nesting level.

    No pipeline field is whitelisted or dropped.  In particular, a changed
    ``pipeline["version"]`` always changes the digest.  Dict insertion order
    never affects the result.

    Args:
        base_source: shader source text.
        key: normalized variant key tuple.
        pipeline: JSON-serializable mapping of pipeline settings, for example
            ``{"version": "2.1", "defines": {"USE_FOG": 1}}``.

    Returns:
        A 64-character lowercase hexadecimal SHA-256 digest.  Equal arguments
        produce equal strings; a different source hash, key or pipeline value
        is designed to produce a different digest.

    Raises:
        ShaderError: if ``pipeline`` is not a mapping or contains a value that
            cannot be serialized deterministically to JSON.
    """
    if not isinstance(base_source, str):
        raise ShaderError("base_source must be a string")
    if not isinstance(pipeline, Mapping):
        raise ShaderError("pipeline must be a mapping")

    source_hash = hashlib.sha256(base_source.encode("utf-8")).hexdigest()
    try:
        # sort_keys recurses into every nested mapping, so dict insertion
        # order anywhere in the pipeline cannot change the digest.
        payload = json.dumps(
            [source_hash, key, pipeline],
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ShaderError("pipeline cannot be serialized deterministically to JSON") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def classify_failure(error: str) -> str:
    """Classify one compiler/driver diagnostic into a retry-policy bucket.

    Matching is case-insensitive.  Buckets are checked in this order:

    * ``"syntax"``: ``syntax``, ``parse error``, ``unexpected token``,
      ``"expected"``;
    * ``"link"``: ``link``, ``undefined reference``, ``unresolved external``,
      ``duplicate symbol``;
    * ``"unsupported"``: ``unsupported``, ``not supported``,
      ``requires feature``, ``extension unavailable``;
    * ``"driver"``: everything else, including unknown text, empty text,
      device loss and out-of-memory reports.

    A syntax failure is therefore never reported as ``"driver"`` and is not
    retried as a transient driver failure.

    Args:
        error: raw diagnostic text.

    Returns:
        Exactly one of ``"syntax"``, ``"link"``, ``"unsupported"`` or
        ``"driver"``.

    Raises:
        ShaderError: if ``error`` is not a string.
    """
    if not isinstance(error, str):
        raise ShaderError("failure diagnostic must be a string")
    text = error.lower()
    for markers, bucket in (
        (_SYNTAX_MARKERS, "syntax"),
        (_LINK_MARKERS, "link"),
        (_UNSUPPORTED_MARKERS, "unsupported"),
    ):
        if any(marker in text for marker in markers):
            return bucket
    return "driver"


def fallback_chain(key: tuple, supported: set) -> list[tuple]:
    """Return the ordered degradation candidates for a variant key.

    The chain starts with the original ``key`` and ends with the empty key
    ``()``, the all-defaults fallback.  Intermediate candidates are produced by
    dropping one feature at a time from the end of the normalized key (the
    alphabetically greatest feature is treated as the least important), then
    keeping only candidates present in ``supported``.

    The original is kept even when it is unsupported so the caller can report
    it.  Every other candidate must be in ``supported``.  The empty key must be
    supported because it is the last-resort fallback; for a non-empty key that
    makes the returned chain at least two entries long.

    Args:
        key: normalized variant key tuple.
        supported: set of key tuples that are known to compile.

    Returns:
        A list of key tuples with ``chain[0] == key`` and
        ``chain[-1] == ()``.  Intermediate entries are supported degradation
        candidates in drop order.

    Raises:
        ShaderError: ``key`` is not a hashable normalized tuple, or ``()`` is
            not in ``supported`` (there is then no safe fallback to return).
    """
    if not isinstance(key, tuple):
        raise ShaderError("variant key must be a normalized tuple")
    try:
        hash(key)
    except TypeError as exc:
        raise ShaderError("variant key must be hashable") from exc
    try:
        if () not in supported:
            raise ShaderError("the empty fallback variant () must be supported")
    except TypeError as exc:
        raise ShaderError("supported must be a set of variant key tuples") from exc

    chain = [key]
    # Drop the trailing (least important) feature one at a time.
    for cut in range(len(key) - 1, -1, -1):
        candidate = key[:cut]
        if candidate in supported:
            chain.append(candidate)
    return chain


def _variant_identity(variant: dict) -> tuple:
    """Return the content identity ``(normalized key, source_hash)``."""
    if not isinstance(variant, Mapping):
        raise ShaderError("each variant must be a dictionary")
    if "key" not in variant or "source_hash" not in variant:
        raise ShaderError("variant requires both 'key' and 'source_hash' fields")
    try:
        normalized = normalize_key(dict(variant["key"]))
    except (TypeError, ShaderError) as exc:
        raise ShaderError("variant 'key' must be a normalized key tuple") from exc
    return normalized, variant["source_hash"]


def dedupe_variants(variants: list[dict]) -> list[dict]:
    """Remove duplicate variants by cache identity, preserving first order.

    A variant is the inline dict ``{"key": tuple, "source_hash": str}``.  Its
    identity is the pair ``(normalized key, source_hash)`` -- the same content
    that feeds :func:`cache_key` once base source and pipeline are known.  Two
    separately allocated dicts with equal content are duplicates; object
    identity is irrelevant.  The first dict for each identity is kept, and the
    relative order of first occurrences is preserved.

    Args:
        variants: list of variant dicts.

    Returns:
        A new list containing the first occurrence of each distinct identity.

    Raises:
        ShaderError: a variant is not a mapping or lacks the required ``key``
            and ``source_hash`` fields.
    """
    if not isinstance(variants, list):
        raise ShaderError("variants must be a list of variant dictionaries")
    seen = set()
    result = []
    for variant in variants:
        identity = _variant_identity(variant)
        try:
            if identity in seen:
                continue
            seen.add(identity)
        except TypeError as exc:
            raise ShaderError("variant identity is not hashable") from exc
        result.append(variant)
    return result


def select_batch(requests: list[dict], compiled: dict, budget: int) -> dict:
    """Choose a compile batch that fits a bounded budget.

    Requests are processed in input order.  A request is a variant dict with an
    optional positive integer ``cost`` (default ``1``)::

        {"key": tuple, "source_hash": str, "cost": 3}

    ``compiled`` maps ``(key, source_hash)`` pairs to already available
    artifacts.  Cached requests are reported in ``"cached"`` and cost nothing.
    An uncached request is selected only when its full cost fits in the
    remaining budget; otherwise it is reported in ``"skipped"``.  The function
    never substitutes a different key for a skipped request, so every entry in
    ``"compile"`` was explicitly requested by the caller.

    Args:
        requests: list of request dicts in submission order.
        compiled: mapping of ``(key, source_hash)`` to a cached artifact; any
            truthy value counts as available.
        budget: non-negative total compile-cost budget.

    Returns:
        ``{"compile": [...], "cached": [...], "skipped": [...], "used": int}``
        where the lists contain the original request dicts in input order and
        ``used`` is the total cost of the requests in ``"compile"``.

    Raises:
        ShaderError: ``budget`` is negative, a request is malformed, or a cost
            is not a positive integer.
    """
    if not isinstance(requests, list):
        raise ShaderError("requests must be a list of request dictionaries")
    if not isinstance(compiled, Mapping):
        raise ShaderError("compiled must be a mapping of cached artifacts")
    if not isinstance(budget, int) or budget < 0:
        raise ShaderError("budget must be a non-negative integer")

    batch = {"compile": [], "cached": [], "skipped": [], "used": 0}
    remaining = budget
    for request in requests:
        identity = _variant_identity(request)
        cost = request.get("cost", 1)
        if not isinstance(cost, int) or cost <= 0:
            raise ShaderError("variant cost must be a positive integer")

        if compiled.get(identity):
            batch["cached"].append(request)
        elif cost <= remaining:
            batch["compile"].append(request)
            remaining -= cost
            batch["used"] += cost
        else:
            # Never substitute another variant: report the request as skipped.
            batch["skipped"].append(request)
    return batch


def apply_hot_reload(base_source: str, changed_files: list[str], cache: dict) -> dict:
    """Invalidate only cache entries affected by changed source files.

    ``cache`` maps opaque cache keys to entries::

        {
            "<cache key>": {
                "key": tuple,
                "source_hash": str,   # sha256(base_source) when cached
                "dependencies": ("fog.glsl", "common.glsl"),
            },
        }

    An entry is invalidated when either

    * its ``source_hash`` no longer matches ``sha256(base_source)`` (the common
      source itself changed), or
    * one of its ``dependencies`` is named in ``changed_files``.

    All other entries are kept.  Editing one include therefore recompiles only
    the variants that actually consume it instead of the whole cache.  File
    names are compared exactly as given, so callers should use one spelling per
    path.  The cache is not mutated; the caller decides when to evict the
    reported keys.

    Args:
        base_source: current common shader source.
        changed_files: file names that changed in this hot-reload event.
        cache: mapping of cache keys to entry metadata as described above.

    Returns:
        ``{"invalidated": [cache_key, ...], "kept": int}``.  ``invalidated``
        preserves cache insertion order and ``kept`` counts entries that were
        not invalidated.

    Raises:
        ShaderError: an entry is missing required metadata or has a malformed
            dependency list.
    """
    if not isinstance(base_source, str):
        raise ShaderError("base_source must be a string")
    if not isinstance(changed_files, list):
        raise ShaderError("changed_files must be a list of file names")
    changed = set()
    for path in changed_files:
        if not isinstance(path, str):
            raise ShaderError("changed file names must be strings")
        changed.add(path)
    if not isinstance(cache, Mapping):
        raise ShaderError("cache must be a mapping of cache entries")

    current_hash = hashlib.sha256(base_source.encode("utf-8")).hexdigest()
    invalidated = []
    kept = 0
    for cache_entry_key, entry in cache.items():
        if not isinstance(entry, Mapping):
            raise ShaderError("each cache entry must be a metadata mapping")
        for field in ("key", "source_hash", "dependencies"):
            if field not in entry:
                raise ShaderError(f"cache entry is missing required field {field!r}")
        dependencies = entry["dependencies"]
        if isinstance(dependencies, str) or not isinstance(dependencies, (list, tuple)):
            raise ShaderError("cache entry 'dependencies' must be a list or tuple of names")
        for dependency in dependencies:
            if not isinstance(dependency, str):
                raise ShaderError("dependency file names must be strings")

        source_changed = entry["source_hash"] != current_hash
        dependency_changed = any(dependency in changed for dependency in dependencies)
        if source_changed or dependency_changed:
            invalidated.append(cache_entry_key)
        else:
            kept += 1

    return {"invalidated": invalidated, "kept": kept}
