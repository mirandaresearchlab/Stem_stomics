"""
This module mirrors the diffusion factory pattern used in Stem.diffusion but
implements a continuous-time flow-matching objective and a simple Euler sampler.
It is kept separate to avoid touching existing DDPM code paths.
"""

from .flow_matching import FlowMatching, create_flow_matching

__all__ = ["FlowMatching", "create_flow_matching"]
