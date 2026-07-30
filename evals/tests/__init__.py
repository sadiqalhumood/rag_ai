"""Tests for the eval harness itself.

The harness grades `anyrag`; these grade the harness. A wrong nDCG would
silently invalidate every number in the report, so the metrics are pinned
against hand-computed cases rather than against their own output.
"""
