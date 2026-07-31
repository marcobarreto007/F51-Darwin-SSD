from __future__ import annotations

import torch

from f51_darwin.transplant_16b.exact_assembly import (
    _neutralize_first_boot,
    publish_exact_candidate,
)


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ttm_residual_gate = torch.nn.Parameter(torch.ones(()))
        self.block = torch.nn.Module()
        self.block.gaba = torch.nn.Module()
        self.block.gaba.residual_gate = torch.nn.Parameter(torch.ones(()))
        self.block.moe = torch.nn.Module()
        self.block.moe.neuroendocrine = torch.nn.Module()
        self.block.moe.neuroendocrine.baseline_dopamine = (
            torch.nn.Parameter(torch.ones(1))
        )
        self.block.moe.neuroendocrine.register_buffer(
            "dopamine", torch.ones(1)
        )
        self.block.moe.neuroendocrine.register_buffer(
            "cortisol", torch.ones(())
        )


def test_first_boot_neutralization_is_exact() -> None:
    model = _Tiny()
    _neutralize_first_boot(model)
    assert all(
        torch.count_nonzero(value) == 0
        for value in model.state_dict().values()
    )


def test_publication_implementation_is_fail_closed() -> None:
    names = set(publish_exact_candidate.__code__.co_names)
    assert "verify_shard_manifest" in names
    assert "sha256_file" in names
