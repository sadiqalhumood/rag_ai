"""anyrag evaluation harness.

The harness is the judge: it generates its own data, so it knows every gold
answer by direct computation and never has to ask a model -- or the system under
test -- what the right answer is.

Nothing in this package may be imported by `anyrag`. The dependency runs one
way only.
"""

from __future__ import annotations

__all__ = ["gen_db", "questions", "metrics"]
