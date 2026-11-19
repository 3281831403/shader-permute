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


class ShaderError(Exception):
    """Raised for invalid shader-variant input or an impossible request."""


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def classify_failure(error: str) -> str:
    """Classify one compiler/driver diagnostic into a retry-policy bucket.

    Matching is case-insensitive.  Buckets are checked in this order:

    * ``"syntax"``: ``syntax``, ``parse error``, ``unexpected token``,
      ``expected``;
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
