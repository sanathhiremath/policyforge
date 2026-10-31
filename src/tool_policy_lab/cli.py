import argparse
import hashlib
import json
from pathlib import Path

from .environment import ENVIRONMENT_VERSION
from .evaluation import evaluate_models, held_out_scenarios
from .training import save_checkpoint, train


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and independently evaluate bounded tool-recovery policies")
    parser.add_argument("command", choices=["demo"])
    parser.add_argument("--episodes", type=int, default=8000)
    parser.add_argument("--evaluation-count", type=int, default=160)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.evaluation_count < 1:
        parser.error("Episode counts must be positive")
    seeds = (11, 23, 37)
    scenarios = held_out_scenarios(args.evaluation_count)
    models = [train(seed, args.episodes) for seed in seeds]
    evaluated_ids = {scenario.identity for scenario in scenarios}
    for model in models:
        if evaluated_ids.intersection(model["training_scenario_ids"]):
            raise RuntimeError("Evaluation leakage")
        save_checkpoint(model, args.output / "checkpoints" / f"policy-{model['seed']}.json")
    comparisons, runs = evaluate_models(models, scenarios)
    q, heuristic = comparisons[:2]
    trajectories = []
    example = next((scenario for scenario in scenarios if "auth" in scenario.faults and "rate_limit" in scenario.faults and "permanent" not in scenario.faults), scenarios[0])
    for name in ("q-learning", "heuristic", "fixed-retry", "random"):
        trajectories.append(next(run for run in runs if run["policy"] == name and run["scenario"] == example.identity[:12]))
    q_failures = [run for run in runs if run["policy"] == "q-learning" and not run["success"]]
    if q_failures:
        trajectories.append(q_failures[0])
    source_digest = hashlib.sha256(b"".join(file.read_bytes() for file in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()
    artifact = {
        "version": "0.1",
        "summary": {"environment": ENVIRONMENT_VERSION, "training_runs": len(seeds), "episodes_per_seed": args.episodes, "evaluation_episodes": len(scenarios), "learned_success_rate": q["success_rate"], "heuristic_success_rate": heuristic["success_rate"], "conclusion": "The learned policy is compared with a domain heuristic, not just a weak retry baseline. This small known environment does not establish that RL is preferable to a deterministic workflow."},
        "comparisons": comparisons,
        "learning_curves": [{"seed": model["seed"], "points": model["curve"]} for model in models],
        "trajectories": trajectories,
        "per_seed": [{"seed": seed, "success_rate": sum(run["success"] for run in runs if run.get("training_seed") == seed) / len(scenarios)} for seed in seeds],
        "provenance": {"training_seeds": list(seeds), "evaluation_seed": 900001, "random_baseline_seed_rule": "1000000 + first eight scenario SHA256 hex digits", "partition": "SHA256(scenario) modulo5: zero=evaluation; other=training", "evaluation_scenario_ids": sorted(evaluated_ids), "training_scenario_ids": {str(model["seed"]): model["training_scenario_ids"] for model in models}, "source_sha256": source_digest, "reward": "-0.2/action; +1/stage; +10/complete; -8/unauthorized attempt; -2/exhaustion; -0.5/abort; extra premature-call penalties", "algorithm": {"name": "tabular Q-learning", "alpha": 0.2, "gamma": 0.95, "epsilon": "0.8 linearly decays to0.04 by85% of training"}},
        "limitations": ["Synthetic deterministic sandbox with three tool stages and five known failure categories; no LLM or real service is called.", "Fault types and remaining budget are observable; future faults and delays are hidden. Tabular Q values use observations rather than a complete hidden environment state. This is not a language-agent benchmark.", "Evaluation holds out exact parameterized fault sequences, not new fault categories. Training uses weighted fault draws; evaluation uniformly samples the held-out finite scenario pool.", "Permanent-failure scenarios cannot be completed; safe aborts do not count as task successes.", "Q-learning aggregates three checkpoints over identical evaluation scenarios; other baselines run once per scenario. Compare rates, not raw unsafe-action totals.", "Training curves include exploration and are not held-out learning curves; the evaluation split is used only after training.", "Unsafe attempts are rejected inside the sandbox. Zero attempts in this experiment is not a safety guarantee."]
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "demo.json").write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(args.output / "demo.json"), "comparisons": comparisons}, indent=2))
    return 0
