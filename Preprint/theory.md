# SHP — Theory (honest formulation)

> Source for the paper's Theory section. Every claim below is either **proven by
> construction** (and machine-checked in `backend/theory_checks.py`) or marked as
> an **empirical conjecture** that the experiments measure rather than assume.
> We deliberately do **not** claim a Banach contraction; see §Non-claims.

## Setup and notation

A scenario is a tuple $(q, C, g)$: question $q$, retrieved contexts $C$, and
ground-truth answer $g$. The Writer→Critic loop produces a sequence of drafts
$x_1, x_2, \dots$. An embedding map $\phi$ (frozen `BAAI/bge-small-en-v1.5`,
$L_2$-normalised, 384-dim) sends each draft to $e_t = \phi(x_t)$. Define the
per-round **semantic distance**

$$ d_t \;=\; 1 - \cos\!\big(e_t, e_{t-1}\big) \;=\; 1 - \frac{\langle e_t, e_{t-1}\rangle}{\lVert e_t\rVert\,\lVert e_{t-1}\rVert}\in[0,2], \qquad t \ge 2 .$$

A judge assigns four RAGAS metrics $m \in \mathcal{M} = \{\text{faith}, \text{rel}, \text{prec}, \text{rec}\}$, each in $[0,1]$. With weights $w$ on the simplex
$\Delta = \{w : w_m \ge 0,\ \sum_m w_m = 1\}$ the **Information Score** is

$$ \mathrm{IS}_t \;=\; \sum_{m\in\mathcal{M}} w_m \, \mathrm{metric}_m(x_t) \in [0,1]. $$

The **halting operator** $H$ (file `backend/halting.py`) maps the round state
$s_t = (t,\ (d_2,\dots,d_t),\ (\mathrm{IS}_1,\dots,\mathrm{IS}_t),\ \text{feedback}_t)$
to a decision in $\{\textsf{continue}\} \cup \mathcal{R}$, where the reason set is
$\mathcal{R} = \{\text{critic\_approved},\ \text{entropy\_convergence},\ \text{no\_information\_gain},\ \text{failsafe}\}$,
evaluated in that fixed priority order. Signal toggles
$(\beta_{\text{critic}}, \beta_{\text{entropy}}, \beta_{\text{gain}})\in\{0,1\}^3$
enable ablations; the failsafe has no toggle.

## Proven results

**Theorem 1 (Deterministic termination).**
For any scenario, any weights, any signal configuration, and any (possibly
adversarial) draft sequence, the loop halts after at most $T_{\max} = \texttt{MAX\_ROUNDS}$
rounds.

*Proof.* The critic increments $t$ by exactly one per round (`agents.py`,
`critic_node`). The failsafe clause of $H$ returns `failsafe` whenever
$t \ge T_{\max}$, and this clause is unconditional on the other signals and not
governed by any toggle. Hence $H(s_t)\neq\textsf{continue}$ for all $t\ge T_{\max}$,
so the realised stopping time $\tau = \min\{t : H(s_t)\neq\textsf{continue}\}$
satisfies $\tau \le T_{\max}$. $\qquad\blacksquare$

*Machine check.* `assert_termination` drives $H$ with all optional signals
disabled and inputs constructed so every non-failsafe signal says continue
(large $d_t$, strictly increasing $\mathrm{IS}_t$, non-approving feedback);
across 200 randomised probes the failsafe always halts with reason `failsafe`.

This is the honest replacement for the discarded Banach claim: SHP guarantees
*termination*, not *contraction*.

**Lemma 1 (Well-definedness).**
(a) If $w\in\Delta$ and every metric lies in $[0,1]$ then $\mathrm{IS}_t\in[0,1]$
(convex combination). (b) The optimiser output lies on $\Delta$. (c) $d_t$ is a
total function into $[0,2]$, finite for every finite input, with the degenerate
zero-norm case mapped conservatively to $1.0$ so it can never cause a
false-positive halt.

*Machine check.* `assert_is_bounds` (500 probes), `assert_weights_on_simplex`,
`assert_distance_total` (200 probes + the zero-norm case).

**Lemma 2 (Halt-priority consistency).**
The reason reported after a run equals the reason the live cascade produced
during the run. *Proof.* Both the live conditional edge
(`agent_workflow.check_convergence`) and the post-hoc derivation
(`agent_workflow.run_scenario` via `halting.derive_halt_reason`) call the single
function `shp_should_halt` with the same priority order; there is no second copy
of the cascade to disagree. *Machine check.*
`assert_halt_priority_consistency` replays the live cascade round-by-round and
compares to the post-hoc reason on every recorded trajectory. (This closes a
real prior bug: the two code paths previously ordered the failsafe and entropy
checks differently and could report different reasons.)

## Conjecture (empirical — measured, not assumed)

**Conjecture 1 (Semantic non-expansiveness).**
Across trajectories the distance sequence $d_t$ is, on average, non-increasing in
$t$: drafts change less from round to round as the loop proceeds.

We make **no** theoretical guarantee here. `empirical_monotonicity_report`
reports, over all test trajectories: the fraction that are monotone
non-increasing, the mean per-trajectory OLS slope of $d_t$ vs $t$ with a
bootstrap 95% CI, and a one-sided Wilcoxon signed-rank test on the step
differences $d_t - d_{t-1}$. The paper reports whatever these come out to,
including a null result. Conjecture 1 is the *motivation* for the entropy
early-stopping rule; Theorem 1 is what makes the system safe even when
Conjecture 1 fails on a given input.

## Non-claims (what we explicitly do not assert)

- **Not a Banach contraction.** The Writer→Critic update has no established
  Lipschitz constant $<1$ and is not deterministic across API calls (even at
  temperature 0), so a contraction/unique-fixed-point guarantee would be
  unsupported. The patience rule is *semantic early stopping*, in the same
  family as early stopping in ML training.
- **Not a uniqueness result.** Different runs may converge to different drafts;
  we claim termination and an empirically characterised distance trajectory,
  not a unique semantic fixed point.
- **IS is a proxy, not ground truth.** RAGAS metrics come from an LLM judge and
  are treated as a noisy quality proxy; the quality claims are validated by
  non-inferiority testing and judge-reliability analysis, not asserted.
