# Learning-Gain Simulation — Design

Date: 2026-07-27
Status: approved concept; implementation pending written-spec review
Scope: isolated scientific simulation, not Darwin-X integration

## Objective

Test whether selective plasticity can obtain more retained predictive improvement
per unit of real computational cost than indiscriminate learning.

The experiment does not attempt to demonstrate consciousness, biological
realism, general intelligence, or superiority over Transformers. It tests one
smaller proposition:

> A learner that estimates the future value of an update and performs the
> smallest justified state change can achieve higher net learning gain than a
> frozen learner, an always-update learner, and a surprise-only learner.

The simulation must be independent of the active Darwin training process. It
must not load a Darwin checkpoint, use a GPU, contact external services, or
write into a checkpoint root.

## Primary Quantity

For policy \(p\), scenario \(s\), and seed \(k\):

\[
G_{p,s,k} =
\frac{
  \Delta Q_{future}
  \cdot R_{old}
  \cdot X_{unseen}
}{
  1
  + C_{compute}
  + C_{write}
  + C_{interference}
}
\]

where:

- \(\Delta Q_{future}\) is improvement on future, unseen observations relative
  to the frozen policy;
- \(R_{old}\) is retained performance on previously learned regimes, bounded to
  \([0,1]\);
- \(X_{unseen}\) is generalization to held-out variants, bounded to \([0,1]\);
- \(C_{compute}\) counts operations that were actually executed;
- \(C_{write}\) counts and weights persistent state changes;
- \(C_{interference}\) measures damage to previously useful predictions.

All components are also reported separately. The scalar is a ranking aid, not
a substitute for the component metrics. A policy fails if it improves the
scalar only by sacrificing a primary component beyond its gate.

## Experimental Unit

An environment emits an ordered stream of experiences:

\[
e_t = (x_t, c_t, y_t, q_t, d_t)
\]

- \(x_t\): observable feature vector;
- \(c_t\): latent regime used only by the evaluator;
- \(y_t\): outcome the learner must predict;
- \(q_t\): consequence importance;
- \(d_t\): delay before the consequence becomes observable.

The learner maintains three timescales:

- fast state: current prediction and confidence;
- episodic state: bounded recent experience and eligibility traces;
- slow state: persistent regime prototypes and outcome estimates.

The evaluator owns the latent regime and held-out probes. Policies never receive
ground-truth regime identifiers, future outcomes, oracle parameters, or
counterfactual results before deciding whether to update.

## Update Actions

At each eligible observation a policy chooses exactly one action:

1. `skip`: no persistent change;
2. `local_update`: bounded update to an existing prototype/outcome estimate;
3. `allocate`: create a new bounded regime slot;
4. `consolidate`: merge or protect repeatedly useful state.

Every executed action has explicit compute and write costs. A rejected action
must not be executed speculatively. In particular, an expensive candidate
update cannot be calculated and later counted as skipped.

## Policies

### Frozen

Never changes persistent state after a shared warm-up. It supplies the
no-learning baseline.

### Always Update

Applies a local update to every eligible experience and allocates when no
prototype matches. It tests whether more plasticity is automatically better.

### Surprise Only

Updates when predictive surprisal exceeds a threshold. It tests whether
prediction error alone distinguishes useful novelty from noise.

### Gain Adaptive

Estimates:

\[
\widehat{g}_t =
P(recur \mid h_t)
\cdot I_t
\cdot q_t
- \lambda_w C_{write}
- \lambda_i \widehat{C}_{interference}
- \lambda_c C_{compute}
\]

where \(I_t\) is bounded information gain estimated from the learner's own
pre-update state. It selects the cheapest action with positive expected gain.

The policy may use only past observations and current state. It cannot inspect
the evaluator, future stream, latent regime, or oracle model. Thresholds are
fixed before evaluation and shared across scenarios.

## Ten Scenarios

### S01 — Repeated Predictable Experience

A stable regime repeats with low noise.

Expected discriminator: Always Update continues writing; Gain Adaptive learns
once and then skips without losing quality.

### S02 — Rare Useful Novelty

A low-frequency regime recurs after long intervals and materially changes the
outcome.

Expected discriminator: Gain Adaptive allocates and retains the regime from few
examples; Frozen misses it and Surprise Only may require more writes.

### S03 — Surprising Noise

Outliers have high prediction error but do not recur and have low consequence
importance.

Expected discriminator: Surprise Only overlearns; Gain Adaptive rejects
non-recurring novelty.

### S04 — Gradual Drift

The outcome mapping moves continuously over time.

Expected discriminator: Gain Adaptive uses bounded local updates instead of
unbounded allocation or total freezing.

### S05 — Abrupt Regime Change

The old mapping becomes invalid at a hidden change point.

Expected discriminator: Gain Adaptive detects sustained failure and replaces or
forks state quickly enough to recover.

### S06 — Shuffled Temporal Correlation

Marginal feature and outcome frequencies are preserved while temporal pairing
is shuffled.

Expected discriminator: apparent learning gain must collapse. A policy that
still reports causal gain fails this scenario.

### S07 — Delayed Consequence

The outcome arrives after a variable delay and only a subset of prior states
was eligible.

Expected discriminator: eligibility traces improve credit assignment without
updating every preceding state.

### S08 — Conflicting Knowledge

Two contexts share features but require incompatible predictions.

Expected discriminator: Gain Adaptive separates contexts and retains both;
Always Update suffers interference.

### S09 — Limited Energy Budget

The stream contains more eligible updates than the fixed compute/write budget
allows.

Expected discriminator: Gain Adaptive spends its budget on updates with the
largest realized future value.

### S10 — Manipulable Internal Metric

The learner can reduce its internal surprise by lowering confidence or
importance estimates without improving external predictions.

Expected discriminator: the independent evaluator rejects metric manipulation.
No policy receives credit for changing its own score without changing held-out
outcomes.

## Controls and Fairness

- Every policy receives byte-identical streams for a given scenario and seed.
- Warm-up data, initial state capacity, and evaluator probes are identical.
- At least 30 deterministic seeds are required.
- Seeds control all random number generators.
- Policy thresholds are selected on separate development seeds and then frozen.
- The all-knowing oracle is reported only as an external ceiling.
- The random gate is matched to the Gain Adaptive policy's update rate after
  evaluation and is reported as a secondary routing control.
- Evaluator probes are never used for learning or update decisions.
- Actual executed operations determine cost. Estimated or avoided work is not
  counted as executed.
- Scenario generation must be auditable from saved configuration and seed.

## Metrics

Primary:

- future predictive quality;
- retained old-regime quality;
- held-out generalization;
- net learning gain;
- real compute operations;
- persistent writes;
- interference cost.

Diagnostic:

- updates per 1,000 experiences;
- false-update rate on S03 and S06;
- detection and recovery delay on S04 and S05;
- delayed-credit precision on S07;
- per-context retention on S08;
- useful gain per budget unit on S09;
- internal-score versus external-quality divergence on S10;
- calibration error and Brier score;
- number of allocated, merged, protected, and evicted regime slots.

`fraction_correct_from_memory`, routing fraction, and nominal FLOP savings must
not be labelled predictive gain. They may be reported only under their literal
names.

## Falsification Gates

The Gain Adaptive hypothesis is rejected if any of these occurs:

1. median net gain does not exceed Always Update and Surprise Only across the
   ten scenarios;
2. apparent compute saving includes an expensive operation executed before a
   skip decision;
3. S03 or S06 shows improvement over matched controls without recurring or
   temporally valid information;
4. quality falls more than 5 percentage points below the best non-oracle policy
   solely to save compute;
5. retained quality on S08 falls more than 10 percentage points after learning
   the conflicting context;
6. S10 improves internal score without statistically distinguishable external
   improvement;
7. repeated runs with identical seed and configuration differ;
8. conclusions depend on a single seed or only one scenario.

Success requires a paired effect with bootstrap 95% confidence intervals and
the complete component metrics. Passing the scalar alone is insufficient.

## Output Contract

The implementation will emit:

- one immutable JSON configuration;
- one row-per-policy/scenario/seed CSV or JSONL file;
- one aggregate JSON report;
- one Markdown report containing component metrics, confidence intervals,
  failures, and scenario-level verdicts;
- optional plots generated only from the raw result file.

Every artifact records source commit, configuration hash, implementation
version, timestamp, seed list, and exact output schema version.

## Proposed Implementation Structure

The simulator remains outside the production model:

- `research/learning_gain/state.py`: learner state, bounded memory, and update
  actions;
- `research/learning_gain/policies.py`: Frozen, Always Update, Surprise Only,
  Gain Adaptive, and matched-random control;
- `research/learning_gain/scenarios.py`: deterministic generators S01–S10;
- `research/learning_gain/evaluator.py`: hidden ground truth, probes, metrics,
  and leakage checks;
- `research/learning_gain/runner.py`: paired-seed execution and artifact
  assembly;
- `src/scripts/run_learning_gain_sim.py`: CLI only;
- `src/tests/test_learning_gain_sim.py`: invariant and falsification tests.

The supported command will be:

```powershell
python src/scripts/run_learning_gain_sim.py `
  --scenarios all `
  --seeds 30 `
  --output-dir workspace/runtime/learning_gain_sim
```

The command must refuse to use a checkpoint-root path as its output directory.
The research package cannot import the Darwin model, optimizer, serving
runtime, or checkpoint writer.

Data flow is one-way:

```text
immutable config + seed
        -> hidden scenario stream
        -> policy observation
        -> prediction
        -> delayed observable consequence
        -> optional bounded update
        -> independent evaluator probes
        -> raw results
        -> aggregate report
```

## Implementation Boundary

The first implementation is a small CPU-only mathematical simulator. Python and
NumPy may be used as an experimental harness because the claim concerns the
learning law, not the production substrate. No conclusion about runtime energy
efficiency may be drawn from Python wall time.

If the learning law survives the ten scenarios, a later, separately approved
experiment may implement the same state machine in a native/event-driven
runtime and measure energy. That is outside this specification.

No code from `memory_entity_sim.py` or `router_dual_path_sim.py` is assumed
correct. Reuse requires independent justification and characterization tests.

## Error Handling

The run aborts with a non-zero exit code when:

- a seed is not reproducible;
- a policy reads evaluator-only state;
- any metric becomes NaN or infinite;
- an operation is charged differently from what was executed;
- a policy exceeds its memory, compute, or write budget;
- a scenario violates its declared marginal or temporal invariants;
- an output path resolves inside a checkpoint root;
- an artifact schema or configuration hash disagrees within the same run.

Interrupted runs may preserve raw rows but must mark the aggregate status
`incomplete`. An incomplete run cannot emit a PASS verdict.

## Validation

Before scientific execution:

- deterministic replay test;
- evaluator isolation test;
- no-oracle-access test;
- no-speculative-expensive-work test;
- accounting conservation test;
- scenario invariant tests;
- seed-pairing test;
- S06 marginal-preservation test;
- S10 evaluator-independence test.

The scientific run begins only after these tests pass.
