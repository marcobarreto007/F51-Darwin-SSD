"""Shared dependency context for organism responsibility modules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from f51_darwin.brainstem import Brainstem, BrainstemConfig, HomeostasisReport
from f51_darwin.dae_optimizer import DAEHybridOptimizer, build_optimizer
from f51_darwin.darwin_x import (
    DarwinXConfig,
    DarwinXModel,
    DarwinXOutput,
    migrate_mutational_state_for_load,
)
from f51_darwin.darwin_x_training import (
    amp_settings,
    load_token_ids,
    validate_token_source,
)
from f51_darwin.data import CausalLMDataLoader
from f51_darwin.data_factory import DataFactory, DataFactoryPaths
from f51_darwin.data_firewall import DataFirewall, FirewallConfig
from f51_darwin.dataset_layout import (
    CORPUS_WORKSPACE_RELATIVE,
    WorkspacePaths,
    resolve_dataset_path,
    resolve_feast_token_bin,
    resolve_organism_checkpoints_root,
)
from f51_darwin.dataset_states import DatasetStatus, SourceType
from f51_darwin.evolution_score import (
    EvolutionMetrics,
    calculate_evolution_score,
    should_grow,
)
from f51_darwin.expert_pool import (
    ExpertModule,
    ExpertPool,
    ExpertRecord,
    ModuleState,
)
from f51_darwin.grounded_extractor import GroundedExtractor
from f51_darwin.legacy_layers import LayerTier, LegacyConfig, LegacyLayers
from f51_darwin.lineage_tracker import LineageTracker
from f51_darwin.pruning import ablation_decision, mark_dead, quarantine_if_low_score
from f51_darwin.replay_buffer import ReplayBuffer, ReplayExample, forgetting_proxy
from f51_darwin.soul import F51Soul
from f51_darwin.state_identity import backbone_identity, tokenizer_identity
from f51_darwin.tokenizer import F51BPETokenizer
from f51_darwin.training_observability import (
    HeldoutResult,
    MetricChannels,
    append_jsonl_fsync,
    evaluate_fixed_holdout,
    fixed_batch_starts,
    sha256_file,
    split_tail_holdout,
    tracked_worktree_state,
)
from f51_darwin.workspace_migration import mark_runtime_use


ROOT = Path(__file__).resolve().parents[3]
CANONICAL_CONFIG = ROOT / "src" / "configs" / "darwin_x_100m.yaml"
_LEGACY_ID_PREFIX = "darwin-model-core-v1:"

__all__ = [name for name in globals() if not name.startswith("__")]
