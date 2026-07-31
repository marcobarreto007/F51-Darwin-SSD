from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from f51_darwin.organism.bootstrap import _DarwinBootstrapMixin
from f51_darwin.organism.causal_adapters import ExplicitRequestAdapter
from f51_darwin.organism.causal_bus import (
    AblationArm,
    BusDecision,
    OrganCausalBus,
    Phase,
    StepIdentity,
)
from f51_darwin.organism.causal_ledger import CausalLedger, verify_ledger
from f51_darwin.organism.config import DarwinOrganismConfig
from f51_darwin.replay_buffer import ReplayBuffer
from scripts.darwin_organism import DarwinOrganism


class _TinyTrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.head = nn.Linear(8, 32)
        self.forward_calls = 0

    def forward(self, input_ids, labels=None, domain=None, heartbeat=None):
        self.forward_calls += 1
        logits = self.head(self.embedding(input_ids))
        lm_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
        aux_loss = logits.square().mean() * 0.1
        return SimpleNamespace(
            loss=lm_loss + aux_loss,
            lm_loss=lm_loss,
            aux_loss=aux_loss,
            effective_aux_loss=aux_loss,
            heartbeat_stats={},
        )


def _request(
    *,
    request_id: str,
    phase: Phase,
    target: str,
    operation: str,
    subject: str,
    value: float | bool,
) -> dict[str, object]:
    return {
        "intervention_id": request_id,
        "phase": phase.value,
        "target": target,
        "operation": operation,
        "subject": subject,
        "value": value,
        "valid_from_step": 0,
        "valid_through_step": 10,
    }


def _organism(
    tmp_path: Path,
    *,
    mode: str,
    initial_state: dict[str, torch.Tensor],
    requests: list[dict[str, object]] | None = None,
) -> DarwinOrganism:
    organism = DarwinOrganism.__new__(DarwinOrganism)
    organism.root = tmp_path
    organism.cfg = SimpleNamespace(
        organism_name="tiny-causal",
        canary_run_id="tiny-run",
        causal_mode=mode,
        causal_ledger_jsonl="workspace/runtime/organism/causal/events.jsonl",
        blockchain_enabled=False,
        block_size=4,
        batch_size=1,
        seed=51,
        replay_warmup_examples=100,
        replay_ratio=0.0,
        replay_add_every=20,
        replay_sample_size=1,
        eval_every=10,
        grad_clip=1.0,
        accum_steps=1,
        holdout_tokens=0,
    )
    organism.cycle = 1
    organism.total_steps = 0
    organism.base_checkpoint_id = "base-test"
    organism.device = torch.device("cpu")
    organism.amp_dtype = None
    organism.token_ids = np.arange(64, dtype=np.int64) % 32
    organism.token_count = len(organism.token_ids)
    organism.holdout_starts = ()
    organism.replay = ReplayBuffer(capacity=4, seed=51)
    organism.model = _TinyTrainingModel()
    organism.model.load_state_dict(copy.deepcopy(initial_state))
    organism.optimizer = torch.optim.SGD(organism.model.parameters(), lr=0.01)
    organism.soul = SimpleNamespace(heartbeat=lambda status: None)
    organism._causal_requests = list(requests or [])
    organism._configure_causal_runtime()
    return organism


def _parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _assert_bitwise_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def test_causal_mode_defaults_disabled_and_validates_values() -> None:
    assert DarwinOrganismConfig().causal_mode == "disabled"
    assert DarwinOrganismConfig(causal_mode=" SHADOW ").causal_mode == "shadow"

    with pytest.raises(ValueError, match="causal_mode"):
        DarwinOrganismConfig(causal_mode="production")


def test_bootstrap_constructs_local_bus_and_ledger_only_when_enabled(
    tmp_path: Path,
) -> None:
    disabled = SimpleNamespace(
        root=tmp_path,
        cfg=DarwinOrganismConfig(project_root=str(tmp_path)),
    )
    _DarwinBootstrapMixin._configure_causal_runtime(disabled)
    assert disabled.causal_bus is None
    assert disabled.causal_ledger is None

    enabled = SimpleNamespace(
        root=tmp_path,
        cfg=DarwinOrganismConfig(
            project_root=str(tmp_path),
            causal_mode="shadow",
        ),
    )
    _DarwinBootstrapMixin._configure_causal_runtime(enabled)
    assert enabled.causal_bus.arm is AblationArm.SHADOW
    assert enabled.causal_ledger.path.is_relative_to(
        (tmp_path / "workspace").resolve()
    )

    escaped = SimpleNamespace(
        root=tmp_path,
        cfg=DarwinOrganismConfig(
            project_root=str(tmp_path),
            causal_mode="enforce",
            checkpoint_root="",  # falsy → uses causal_ledger_jsonl
            causal_ledger_jsonl="../outside.jsonl",
        ),
    )
    with pytest.raises(ValueError, match="workspace"):
        _DarwinBootstrapMixin._configure_causal_runtime(escaped)


def test_training_calls_three_phases_in_order() -> None:
    class RecordingBus:
        arm = AblationArm.CONTROL

        def __init__(self) -> None:
            self.phases: list[Phase] = []

        def decide(self, identity, phase, context):
            self.phases.append(phase)
            return BusDecision(
                identity=identity,
                phase=phase,
                arm=AblationArm.CONTROL,
                context=context,
            )

        def feedback(self, outcome):
            return ()

        def record_intent(self, identity, decisions):
            return None

    torch.manual_seed(51)
    initial = _TinyTrainingModel().state_dict()
    organism = _organism(
        Path.cwd(), mode="disabled", initial_state=initial
    )
    recording = RecordingBus()
    organism.causal_bus = recording
    organism.causal_ledger = None

    organism._train_cycle(1, {"forgetting": 0.0})

    assert recording.phases == [
        Phase.PRE_LOSS,
        Phase.PRE_BACKWARD,
        Phase.PRE_OPTIMIZER,
    ]


def test_shadow_is_bitwise_equal_to_disabled_and_control_and_has_no_orphans(
    tmp_path: Path,
) -> None:
    torch.manual_seed(51)
    initial = copy.deepcopy(_TinyTrainingModel().state_dict())
    requests = [
        _request(
            request_id="shadow-aux",
            phase=Phase.PRE_LOSS,
            target="LOSS_TERM",
            operation="SET_SCALE",
            subject="aux",
            value=0.0,
        ),
        _request(
            request_id="shadow-scale",
            phase=Phase.PRE_OPTIMIZER,
            target="GRADIENT_GROUP",
            operation="SCALE",
            subject="all",
            value=0.0,
        ),
    ]
    results = {}
    organisms = {}
    for mode in ("disabled", "control", "shadow"):
        path = tmp_path / mode
        organism = _organism(
            path, mode=mode, initial_state=initial, requests=requests
        )
        organism._train_cycle(1, {"forgetting": 0.0})
        results[mode] = _parameters(organism.model)
        organisms[mode] = organism

    _assert_bitwise_equal(results["disabled"], results["control"])
    _assert_bitwise_equal(results["disabled"], results["shadow"])
    verification = verify_ledger(organisms["shadow"].causal_ledger.path)
    assert verification.valid
    assert verification.orphan_intent_count == 0
    assert verification.event_count == 2
    control_verification = organisms["control"].causal_ledger.verify()
    assert control_verification.valid
    assert control_verification.event_count == 2
    assert control_verification.orphan_intent_count == 0


@pytest.mark.parametrize(
    ("intervention_request", "unchanged_names", "changed_names"),
    [
        (
            _request(
                request_id="skip-update",
                phase=Phase.PRE_BACKWARD,
                target="UPDATE",
                operation="SKIP",
                subject="optimizer",
                value=True,
            ),
            {"embedding.weight", "head.weight", "head.bias"},
            set(),
        ),
        (
            _request(
                request_id="scale-all-zero",
                phase=Phase.PRE_OPTIMIZER,
                target="GRADIENT_GROUP",
                operation="SCALE",
                subject="all",
                value=0.0,
            ),
            {"embedding.weight", "head.weight", "head.bias"},
            set(),
        ),
        (
            _request(
                request_id="scale-head-zero",
                phase=Phase.PRE_OPTIMIZER,
                target="GRADIENT_GROUP",
                operation="SCALE",
                subject="head.weight",
                value=0.0,
            ),
            {"head.weight"},
            {"embedding.weight"},
        ),
    ],
)
def test_enforce_executes_skip_and_gradient_scale(
    tmp_path: Path,
    intervention_request: dict[str, object],
    unchanged_names: set[str],
    changed_names: set[str],
) -> None:
    torch.manual_seed(51)
    initial = copy.deepcopy(_TinyTrainingModel().state_dict())
    organism = _organism(
        tmp_path,
        mode="enforce",
        initial_state=initial,
        requests=[intervention_request],
    )

    organism._train_cycle(1, {"forgetting": 0.0})
    result = _parameters(organism.model)

    for name in unchanged_names:
        assert torch.equal(initial[name], result[name]), name
    for name in changed_names:
        assert not torch.equal(initial[name], result[name]), name
    assert organism.causal_ledger.verify().orphan_intent_count == 0


def test_enforce_executes_aux_scale_and_grad_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(51)
    initial = copy.deepcopy(_TinyTrainingModel().state_dict())
    requests = [
        _request(
            request_id="zero-aux",
            phase=Phase.PRE_LOSS,
            target="LOSS_TERM",
            operation="SET_SCALE",
            subject="aux",
            value=0.0,
        ),
        _request(
            request_id="clip",
            phase=Phase.PRE_OPTIMIZER,
            target="GRAD_CLIP",
            operation="SET_MAX_NORM",
            subject="global",
            value=0.25,
        ),
    ]
    organism = _organism(
        tmp_path, mode="enforce", initial_state=initial, requests=requests
    )
    observed_max_norms: list[float] = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def recording_clip(parameters, max_norm, *args, **kwargs):
        observed_max_norms.append(float(max_norm))
        return original_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)

    organism._train_cycle(1, {"forgetting": 0.0})

    assert observed_max_norms == [0.25]
    events = [
        json.loads(line)
        for line in organism.causal_ledger.path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    outcome = next(
        event for event in events if event["event_type"] == "STEP_OUTCOME"
    )
    assert outcome["payload"]["raw_losses"]["aux"] > 0.0
    assert outcome["payload"]["effective_losses"]["aux"] == 0.0


def test_blocked_decision_prevents_backward_and_optimizer_mutation(
    tmp_path: Path,
) -> None:
    class RaisingAdapter:
        adapter_id = "raising"
        version = "v1"

        def observe(self, identity, phase, context):
            raise RuntimeError("closed")

        def propose(self, identity, phase, signals, context):
            return ()

        def feedback(self, outcome):
            return None

    torch.manual_seed(51)
    initial = copy.deepcopy(_TinyTrainingModel().state_dict())
    organism = _organism(
        tmp_path, mode="enforce", initial_state=initial
    )
    organism.causal_bus = OrganCausalBus(
        arm=AblationArm.APPLY,
        adapters=(RaisingAdapter(),),
        intent_recorder=organism.causal_ledger,
    )

    organism._train_cycle(1, {"forgetting": 0.0})

    _assert_bitwise_equal(initial, _parameters(organism.model))
    assert organism.model.forward_calls == 0
    verification = organism.causal_ledger.verify()
    assert verification.valid
    assert verification.orphan_intent_count == 0


def test_explicit_adapter_is_training_only_and_uses_json_requests() -> None:
    adapter = ExplicitRequestAdapter()
    identity = StepIdentity(
        run_id="run",
        cycle=1,
        optimizer_step=0,
        accumulation_window=0,
        batch_digest="batch",
        rng_digest="rng",
        base_checkpoint_id="base",
        training_contract_id="contract",
    )
    structural = _request(
        request_id="forbidden",
        phase=Phase.CYCLE_BOUNDARY,
        target="STRUCTURAL_ACTION",
        operation="QUEUE",
        subject="checkpoint",
        value=True,
    )

    assert tuple(
        adapter.propose(
            identity,
            Phase.CYCLE_BOUNDARY,
            (),
            {"requests": [structural]},
        )
    ) == ()
    assert vars(adapter) == {}


def test_intent_is_durable_before_model_forward(tmp_path: Path) -> None:
    torch.manual_seed(51)
    initial = copy.deepcopy(_TinyTrainingModel().state_dict())
    organism = _organism(
        tmp_path, mode="enforce", initial_state=initial
    )

    class IntentCheckingModel(_TinyTrainingModel):
        def forward(self, *args, **kwargs):
            verification = organism.causal_ledger.verify()
            assert verification.event_count == 1
            assert verification.orphan_intent_count == 1
            return super().forward(*args, **kwargs)

    checking_model = IntentCheckingModel()
    checking_model.load_state_dict(initial)
    organism.model = checking_model
    organism.optimizer = torch.optim.SGD(
        checking_model.parameters(), lr=0.01
    )

    organism._train_cycle(1, {"forgetting": 0.0})

    assert checking_model.forward_calls == 1
    verification = organism.causal_ledger.verify()
    assert verification.event_count == 2
    assert verification.orphan_intent_count == 0


@pytest.mark.parametrize(
    ("failure_kind", "expected_error"),
    [
        ("nonfinite", "nonfinite_or_missing_loss"),
        ("brainstem", "brainstem_rejected"),
        ("forward", "forward_error:RuntimeError"),
    ],
)
def test_rejected_attempts_close_outcome_without_orphan(
    tmp_path: Path,
    failure_kind: str,
    expected_error: str,
) -> None:
    torch.manual_seed(51)
    initial = copy.deepcopy(_TinyTrainingModel().state_dict())
    organism = _organism(
        tmp_path, mode="enforce", initial_state=initial
    )

    if failure_kind == "nonfinite":
        class NonfiniteModel(_TinyTrainingModel):
            def forward(self, *args, **kwargs):
                output = super().forward(*args, **kwargs)
                output.loss = output.loss * output.loss.new_tensor(
                    float("nan")
                )
                return output

        organism.model = NonfiniteModel()
        organism.model.load_state_dict(initial)
    elif failure_kind == "forward":
        class RaisingForwardModel(_TinyTrainingModel):
            def forward(self, *args, **kwargs):
                self.forward_calls += 1
                raise RuntimeError("forward closed")

        organism.model = RaisingForwardModel()
        organism.model.load_state_dict(initial)
    else:
        organism.brainstem = SimpleNamespace(
            check_loss=lambda value: SimpleNamespace(
                alive=False, violations=["loss_guard"]
            )
        )
    organism.optimizer = torch.optim.SGD(
        organism.model.parameters(), lr=0.01
    )

    if failure_kind == "forward":
        with pytest.raises(RuntimeError, match="forward closed"):
            organism._train_cycle(1, {"forgetting": 0.0})
    else:
        organism._train_cycle(1, {"forgetting": 0.0})

    verification = organism.causal_ledger.verify()
    assert verification.valid
    assert verification.event_count == 2
    assert verification.orphan_intent_count == 0
    events = [
        json.loads(line)
        for line in organism.causal_ledger.path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert expected_error in events[1]["payload"]["error"]
