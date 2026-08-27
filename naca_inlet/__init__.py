"""NACA submerged inlet tool: cut a parametric flush inlet into a STEP surface."""

from .geometry import InletSpec, resolve_spec, spec_summary

__all__ = ["InletSpec", "resolve_spec", "spec_summary"]
__version__ = "1.0.0"
