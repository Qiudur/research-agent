"""Optional source-tree facade; public v1 API lives in library.skills."""

from library.skills import demote_skill, load_skill_verified, promote_candidates, promote_to_skill, verified_wins

__all__ = ["promote_candidates", "promote_to_skill", "load_skill_verified",
           "demote_skill", "verified_wins"]
