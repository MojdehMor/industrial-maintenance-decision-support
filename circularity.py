"""Conditional circularity re-ranking for research evaluation only.

All action taxonomies are provided by the caller. Re-ranking is a score-based
research policy, NOT proof of technical compatibility or safe substitution.
"""
from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence


def conditional_circularity_rerank(
    ranked: Sequence[str],
    scores: Mapping[str, float],
    *,
    replacement_actions: Collection[str],
    preservation_actions: Collection[str],
    delta: float = 0.05,
    pool_size: int = 5,
) -> list[str]:
    """Promote close-scoring preservation actions when rank 1 is replacement.

    Eligible alternatives must fall within the first `pool_size` ranks and
    have a relative score difference from rank 1 <= `delta`. A zero or
    negative top score never triggers promotion. The returned order remains a
    recommendation ordering, not a verified maintenance or procurement action.
    """
    if not 0 <= delta <= 1:
        raise ValueError("delta must be between 0 and 1")
    if pool_size < 2:
        raise ValueError("pool_size must be at least 2")
    original = list(ranked)
    replace = set(replacement_actions)
    preserve = set(preservation_actions)
    if replace & preserve:
        raise ValueError("Replacement and preservation action sets overlap")
    if not original or original[0] not in replace:
        return original
    top = float(scores.get(original[0], 0.0))
    if top <= 0:
        return original
    promoted = [action for action in original[1:pool_size]
                if action in preserve
                and (top - float(scores.get(action, 0.0))) / top <= delta + 1e-12]
    chosen = set(promoted)
    return promoted + [action for action in original if action not in chosen]
