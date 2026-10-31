"""Outcome evaluation kept separate from the optimization reward."""
from itertools import product
import random
from statistics import mean

from .environment import ACTIONS, FAULTS, Scenario, ToolEnvironment
from .training import greedy


def heuristic(observation: str) -> str:
    category = observation.split("|")[0]
    return {"ready": "call", "transient": "wait", "rate_limit": "wait", "auth": "refresh", "permanent": "abort"}[category]


def play(scenario, policy, name: str):
    env = ToolEnvironment(scenario)
    steps, total = [], 0
    while not env.done:
        observation = env.observation
        action = policy(observation)
        next_observation, reward, _ = env.step(action)
        total += reward
        steps.append({"step": len(steps) + 1, "observation": observation, "action": action, "reward": reward, "next_observation": next_observation, "completed": env.completed, "remaining_budget": env.remaining})
    success = env.completed == len(scenario.faults) and env.unsafe_actions == 0
    assert len(steps) <= scenario.budget and env.remaining >= 0
    return {"policy": name, "scenario": scenario.identity[:12], "faults": list(scenario.faults), "delays": list(scenario.delays), "outcome": env.outcome, "success": success, "reward": total, "unsafe_actions": env.unsafe_actions, "budget_exhausted": env.budget_exhausted, "steps": steps}


def held_out_scenarios(count: int = 160, seed: int = 900001):
    population = [Scenario(faults, delays) for faults in product(FAULTS, repeat=3) for delays in product((1, 2), repeat=3)]
    population = [scenario for scenario in population if scenario.partition == "evaluation"]
    if count < 1 or count > len(population):
        raise ValueError(f"Evaluation count must be between1 and {len(population)}")
    random.Random(seed).shuffle(population)
    return population[:count]


def evaluate_models(models: list[dict], scenarios: list):
    runs = []
    for model in models:
        for scenario in scenarios:
            runs.append(play(scenario, lambda obs: greedy(model["q"], obs), "q-learning") | {"training_seed": model["seed"]})
    for name in ("heuristic", "fixed-retry", "random"):
        for scenario in scenarios:
            rng = random.Random(1000000 + int(scenario.identity[:8], 16))
            policy = heuristic if name == "heuristic" else (lambda obs: "call") if name == "fixed-retry" else (lambda obs: rng.choice(ACTIONS))
            runs.append(play(scenario, policy, name))
    comparisons = []
    for name in ("q-learning", "heuristic", "fixed-retry", "random"):
        subset = [run for run in runs if run["policy"] == name]
        comparisons.append({"policy": name, "episodes": len(subset), "success_rate": mean(run["success"] for run in subset), "mean_steps": mean(len(run["steps"]) for run in subset), "unsafe_actions": sum(run["unsafe_actions"] for run in subset), "unsafe_action_rate": mean(run["unsafe_actions"] > 0 for run in subset), "budget_exhaustions": sum(run["budget_exhausted"] for run in subset), "mean_reward": mean(run["reward"] for run in subset)})
    return comparisons, runs
