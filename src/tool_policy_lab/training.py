"""Tabular Q-learning with reproducible exploration and portable checkpoints."""
import json
import random
from pathlib import Path

from .environment import ACTIONS, ToolEnvironment, draw_scenario


def greedy(q: dict, observation: str) -> str:
    values = q.get(observation, [0.0] * len(ACTIONS))
    return ACTIONS[max(range(len(ACTIONS)), key=lambda index: values[index])]


def train(seed: int, episodes: int = 8000):
    if episodes < 1:
        raise ValueError("Need at least one training episode")
    rng = random.Random(seed)
    q = {}
    curve = []
    rewards, successes = [], []
    scenario_ids = set()
    for episode in range(1, episodes + 1):
        scenario = draw_scenario(rng, "training")
        scenario_ids.add(scenario.identity)
        env = ToolEnvironment(scenario)
        epsilon = max(0.04, 0.8 * (1 - episode / (episodes * 0.85)))
        total = 0
        while not env.done:
            observation = env.observation
            values = q.setdefault(observation, [0.0] * len(ACTIONS))
            action = rng.choice(ACTIONS) if rng.random() < epsilon else greedy(q, observation)
            successor, reward, done = env.step(action)
            target = reward if done else reward + 0.95 * max(q.get(successor, [0.0] * len(ACTIONS)))
            index = ACTIONS.index(action)
            values[index] += 0.2 * (target - values[index])
            total += reward
        rewards.append(total)
        successes.append(env.outcome == "completed")
        if episode % 200 == 0 or episode == episodes:
            curve.append({"episode": episode, "mean_reward": sum(rewards[-200:]) / len(rewards[-200:]), "success_rate": sum(successes[-200:]) / len(successes[-200:])})
    return {"seed": seed, "episodes": episodes, "q": q, "curve": curve, "training_scenario_ids": sorted(scenario_ids)}


def save_checkpoint(result: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": "0.1", "seed": result["seed"], "episodes": result["episodes"], "actions": ACTIONS, "q": result["q"]}, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_checkpoint(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["version"] != "0.1" or tuple(value["actions"]) != ACTIONS:
        raise ValueError("Incompatible checkpoint")
    return value["q"]
