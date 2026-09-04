# Memory Audit Tool — Change Log

Working directory: `audit/`
Tool file: `memory_audit_tool.py`
Open Web UI plugin contract preserved (`Tools` class, `Valves`/`UserValves` pydantic
models, async tools with `__user__` / `__request__` / `__event_emitter__`).

---

## v3.3.0 — Fragment / containment duplicate detection

**Problem addressed:**
The audit tool was not flagging short memories that are *fragments* of larger
parent memories. Root cause: Jaccard similarity is structurally biased against
fragments.

Jaccard is computed over meaningful tokens and normalized by the **union**:

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

A fragment is a **subset** of its parent, so the union is dominated by the
parent's tokens and the score collapses to roughly `shared / parent_tokens`.
A single-token fragment of a 3-token parent scores `1 / 4 = 0.25`, which falls
under the default `audit_duplicate_threshold` of `0.28` and is silently missed.

**What changed:**

- Added `_shared_token_count(a, b)` — size of the intersection of two token sets.
- Added `_containment_score(a, b)` — fraction of the *smaller* token set that is
  contained in the larger one. A 1-token fragment of a 3-token parent scores
  `1.0` here (the inverse of the Jaccard case above).
- Rewrote `_find_duplicate_candidates` to combine two signals:
  1. Jaccard overlap (existing behaviour).
  2. New containment / fragment signal, gated by shared-token count.
  Each candidate is now a `(memory, score, is_fragment)` tuple.
- Wired the fragment signal into `_audit_single_memory`: fragment links and
  plain-overlap links are reported separately with distinct notes.
- Bumped `version` to `3.3.0`.

**New user valves (defaults):**

| Valve | Default | Purpose |
|---|---|---|
| `fragment_containment_threshold` | `0.9` | Containment score at/above which a smaller memory is flagged as a fragment of a larger one. |
| `min_shared_tokens_for_fragment` | `2` | Minimum shared meaningful tokens required to flag a fragment — prevents single common-word false positives. |

**How to tune (if needed):**

- Raise `fragment_containment_threshold` toward `1.0` to be stricter (fewer
  fragment links).
- Raise `min_shared_tokens_for_fragment` to require more overlap before flagging
  a fragment (fewer false positives at the cost of catching only heavier ones).
- Lower `audit_duplicate_threshold` to catch more loose Jaccard matches (blunt
  instrument — use with care).

**Testing:** see `test_fragment_detection.py` in this folder.

---

## v3.2.4 — Open Web UI v0.9 compatibility

- Async conversion of memory tool methods (existing baseline).
