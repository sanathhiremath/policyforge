# PolicyForge

Version 0.1.

A working local reinforcement-learning experiment for bounded tool recovery. It trains actual tabular Q-learning policies, saves their checkpoints, and evaluates them on held-out fault sequences alongside a sensible heuristic, fixed retries, and random actions.

This is a small synthetic environment, not an LLM agent, a production service, or evidence of frontier-scale post-training. Its useful engineering question is whether learned recovery justifies its complexity when a straightforward workflow is available.

## Run on Windows or Ubuntu

Python 3.11+ is sufficient; the runtime uses only the standard library.

```bash
python run.py demo
```

The command trains three policies for 8,000 episodes each and evaluates each checkpoint on the same 160 distinct held-out scenarios. It writes `artifacts/demo.json` and `artifacts/checkpoints/policy-{11,23,37}.json`. Paths are resolved from the project location, so running the command from another working directory is supported.

For isolated tests:

```bash
python -m venv .venv
```

Windows: `.venv\Scripts\python -m pip install -e ".[dev]"` and `.venv\Scripts\python -m pytest`.

Ubuntu: `.venv/bin/python -m pip install -e '.[dev]'` and `.venv/bin/python -m pytest`.

## Environment and policy contract

An episode contains three tool stages with a fourteen-action budget. Each stage begins ready, transiently unavailable, rate limited, needing credential refresh, or permanently unavailable. An observation exposes only the current failure category, remaining budget, and number of completed stages. Future failures and cooldown lengths are hidden.

Actions are `call`, `wait`, `refresh`, and `abort`. A successful ready call advances work. Waiting clears transient/rate-limit faults after a hidden one- or two-step cooldown. Refresh clears expired authentication. Calling with expired authentication is rejected by the sandbox and ends the episode. Permanent faults cannot be completed. No action touches an external system.

The environment enforces the budget and rejects actions after termination. Q-learning sees observation-level states; the hidden cooldown means these observations are not a complete Markov state. This is an explicit limitation rather than a claim that tabular Q-learning solves general partially observable planning.

```text
scenario → sandbox environment → observation → learned/baseline action
                                ↓
                    reward for training; outcome for evaluation
                                ↓
                 checkpoint + replayable trajectory + report
```

## Reward and independent success check

Every action costs 0.2 reward. A completed stage adds 1 and full completion adds 10. Rejected authentication attempts cost another 8, exhausted budgets cost 2, and abort costs another 0.5. Premature calls incur extra penalties. The full definition is in `environment.py` and the generated provenance.

The evaluator does **not** equate reward with success. It checks that all stages completed without an unauthorized attempt. Aborting an impossible task may be sensible, but it is not counted as success. The tests include an abort-all counterexample: low action cost does not mean completed work.

## Leakage boundary and reproducibility

Each parameterized scenario has a SHA-256 identity including its full fault sequence, delays, and budget. Identities whose first eight hex digits modulo five equal zero are reserved for evaluation; all other identities are eligible for training. The training code cannot draw from the evaluation partition. Seeds are 11, 23, and 37; the evaluation sampling seed is 900001.

The evaluation pool samples uniformly over distinct held-out combinations; training uses weighted fault draws. The report preserves all sampled identity hashes and verifies no training/evaluation overlap. Holding out combinations within known simulator rules does not establish generalization to novel errors or real tools.

Learning curves are rolling training windows **with exploration**, not repeated evaluations of the held-out set. The saved report includes the source digest, Q-learning parameters, every training scenario identity, evaluation identities, and one same-scenario trajectory per policy. Each training run is also reported separately.

## Measured reference result

The checked-in default run produced:

| Policy | Completion rate | Mean actions | Rejected auth-attempt rate |
| --- | ---: | ---: | ---: |
| Q-learning, three-seed aggregate | 53.75% | 6.60 | 0.00% |
| Domain heuristic | 53.75% | 4.54 | 0.00% |
| Fixed retry | 27.50% | 7.54 | 36.25% |
| Random | 1.88% | 3.73 | 7.50% |

The learned policies matched the heuristic's completion rate, used more actions, and earned lower mean reward. **This experiment does not justify replacing the heuristic with RL.** Permanent-failure scenarios are included in the denominator, which explains why even the heuristic does not complete every task. Raw unsafe-action counts have different denominators: Q-learning evaluates three checkpoints, while each baseline runs once per scenario. Compare rates.

## Files and portfolio use

- `environment.py`: versioned sandbox rules and deterministic scenario partition.
- `training.py`: actual Q updates, exploration, and checkpoint persistence/loading.
- `evaluation.py`: outcome-based scoring and baseline comparisons.
- `cli.py`: experiment orchestration, provenance, and artifact export.
- `tests/`: budget/termination, rejected actions, dataset separation, reproducibility, checkpoint reload, and baseline checks.
- `artifacts/demo.json`: recorded report consumed by the portfolio; viewing it does not train a policy in the browser.

The portfolio should show the heuristic comparison prominently. A useful technical decision is sometimes that a learned policy adds no benefit in a known small environment.

## Repository

Source: https://github.com/sanathhiremath/policyforge


This repository contains a local working prototype. Reproduction commands, measured scope, and limitations are documented above.
