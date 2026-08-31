"""
Smoke test for the fragment-detection change in memory_audit_tool.py.

Imports the Tools class directly and exercises the scoring helpers and
_find_duplicate_candidates with synthetic token lists that mirror the
fragment cases Jaccard misses.

Run:  cd audit && python3 test_fragment_detection.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_audit_tool import Tools


def make_tool():
    t = Tools()
    # Force the new valve defaults.
    t.user_valves.fragment_containment_threshold = 0.9
    t.user_valves.min_shared_tokens_for_fragment = 2
    return t


def test_containment_score():
    t = make_tool()
    # Parent: 5 meaningful tokens. Fragment: 1 of those tokens.
    # Jaccard = 1/5 = 0.2 (< 0.28 threshold) -> missed by Jaccard alone.
    parent = ["developing", "assignments", "concurrent", "python", "software"]
    fragment = ["developing"]
    jaccard = t._jaccard_score(parent, fragment)
    containment = t._containment_score(parent, fragment)
    print(f"single-token fragment: jaccard={jaccard:.3f} containment={containment:.3f}")
    assert jaccard < 0.28, "jaccard should be low (missed by Jaccard)"
    assert containment == 1.0, "containment should be 1.0"


def test_single_token_fragment_not_flagged():
    t = make_tool()
    # Single-token fragment shares only 1 token -> below min_shared=2 -> not flagged.
    parent = {
        "id": "P",
        "tokens": ["developing", "assignments", "concurrent", "python", "software"],
    }
    fragment = {"id": "F", "tokens": ["developing"]}
    candidates = t._find_duplicate_candidates(
        fragment, [parent], threshold=0.28, top_k=1
    )
    print(f"single-token fragment -> candidates: {candidates}")
    assert len(candidates) == 0, "single shared token should NOT be flagged (guard)"
    print("PASS: single shared token correctly not flagged (min_shared guard)")


def test_two_shared_tokens():
    t = make_tool()
    parent = {
        "id": "P",
        "tokens": ["developing", "assignments", "concurrent", "python", "software"],
    }
    fragment = {
        "id": "F",
        "tokens": ["developing", "python"],
    }
    candidates = t._find_duplicate_candidates(
        fragment, [parent], threshold=0.28, top_k=1
    )
    print(f"two-token fragment -> candidates: {candidates}")
    assert len(candidates) == 1, "two-token fragment should be detected"
    _, _, is_fragment = candidates[0]
    assert is_fragment is True
    print("PASS: two-token fragment detected via containment")


def test_no_false_positive_single_word():
    t = make_tool()
    # Parent has the shared word, but fragment is only 1 shared token.
    parent = {
        "id": "P",
        "tokens": ["developing", "assignments", "concurrent", "python"],
    }
    fragment = {"id": "F", "tokens": ["python"]}
    candidates = t._find_duplicate_candidates(
        fragment, [parent], threshold=0.28, top_k=1
    )
    print(f"single shared word -> candidates: {candidates}")
    # min_shared_tokens_for_fragment=2 should suppress this.
    if candidates:
        _, _, is_fragment = candidates[0]
        assert is_fragment is False, "single shared word should NOT be a fragment"
    print("PASS: single shared word not falsely flagged as fragment")


def test_distinct_memories_still_no_link():
    t = make_tool()
    a = {"id": "A", "tokens": ["galaxy", "telescope", "observation"]}
    b = {"id": "B", "tokens": ["recipe", "pasta", "tomato"]}
    candidates = t._find_duplicate_candidates(
        a, [b], threshold=0.28, top_k=1
    )
    print(f"distinct -> candidates: {candidates}")
    assert len(candidates) == 0, "distinct memories should not link"
    print("PASS: distinct memories correctly not linked")


if __name__ == "__main__":
    test_containment_score()
    test_single_token_fragment_not_flagged()
    test_two_shared_tokens()
    test_no_false_positive_single_word()
    test_distinct_memories_still_no_link()
    print("\nALL TESTS PASSED")
