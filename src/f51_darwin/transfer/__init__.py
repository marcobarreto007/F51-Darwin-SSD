"""Transformer-to-SSD architecture transfer for the Darwin organism."""

from .donor import DONOR_MODEL_ID, DonorManifest
from .ssd_mixer import DiscreteSSDMixer

__all__ = ["DONOR_MODEL_ID", "DiscreteSSDMixer", "DonorManifest"]
