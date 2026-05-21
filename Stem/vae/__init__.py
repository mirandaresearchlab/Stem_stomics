"""
Conditional VAE model for spot-level gene expression prediction.

Kept separate from Stem.models (DiT backbone) and Stem.fm (flow matching) so the
three generative paradigms — diffusion, flow matching, VAE — sit side by side
without touching each other's code paths.
"""

from .cvae import CVAE, CVAE_models, ConditionalVAE, cvae_loss

__all__ = ["CVAE", "CVAE_models", "ConditionalVAE", "cvae_loss"]
