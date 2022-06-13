"""This module is the implementation of a Vanilla Variational Autoencoder
(https://arxiv.org/abs/1312.6114).

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:
"""

from .vae_config import VAEConfig
from .vae_model import VAE
from .vae_model_adapted import my_VAE
__all__ = ["VAE", "VAEConfig", "my_VAE"]
