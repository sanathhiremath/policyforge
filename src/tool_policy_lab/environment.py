"""Deterministic sandbox tool environment. No network or real side effects."""
from dataclasses import dataclass
import hashlib
import random

ENVIRONMENT_VERSION = "tool-recovery-0.1"
ACTIONS = ("call", "wait", "refresh", "abort")
FAULTS = ("ready", "transient", "rate_limit", "auth", "permanent")


@dataclass(frozen=True)
class Scenario:
    faults: tuple[str, ...]
    delays: tuple[int, ...]
    budget: int = 14

    def __post_init__(self):
        if len(self.faults) != len(self.delays) or not self.faults:
            raise ValueError("Every task needs a fault and recovery delay")
        if any(fault not in FAULTS for fault in self.faults):
            raise ValueError("Unknown fault")
        if any(delay not in (1, 2) for delay in self.delays) or self.budget < 1:
            raise ValueError("Unsupported delay or budget")

    @property
    def identity(self):
        text = ",".join(self.faults) + ":" + ",".join(map(str, self.delays)) + f":{self.budget}"
        return hashlib.sha256(text.encode()).hexdigest()

    @property
    def partition(self):
        return "evaluation" if int(self.identity[:8], 16) % 5 == 0 else "training"


def draw_scenario(rng: random.Random, partition: str) -> Scenario:
    if partition not in ("training", "evaluation"):
        raise ValueError("Unknown partition")
    while True:
        faults = tuple(rng.choices(FAULTS, weights=(3, 3, 2, 2, 1), k=3))
        scenario = Scenario(faults, tuple(rng.choice((1, 2)) for _ in range(3)))
        if scenario.partition == partition:
            return scenario


class ToolEnvironment:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.completed = 0
        self.remaining = scenario.budget
        self.unsafe_actions = 0
        self.budget_exhausted = False
        self.done = False
        self.outcome = "running"
        self.category = scenario.faults[0]
        self.cooldown = scenario.delays[0]

    @property
    def observation(self):
        # Future faults and recovery delays are hidden from every policy.
        return f"{self.category}|{self.remaining}|{self.completed}"

    def step(self, action: str):
        if self.done:
            raise RuntimeError("Cannot act after terminal outcome")
        if action not in ACTIONS:
            raise ValueError("Unknown action")
        self.remaining -= 1
        reward = -0.2
        if action == "abort":
            self.done, self.outcome = True, "aborted"
            reward -= 0.5
        elif action == "call" and self.category == "ready":
            self.completed += 1
            reward += 1
            if self.completed == len(self.scenario.faults):
                self.done, self.outcome = True, "completed"
                self.category = "terminal"
                reward += 10
            else:
                self.category = self.scenario.faults[self.completed]
                self.cooldown = self.scenario.delays[self.completed]
        elif action == "call" and self.category == "auth":
            self.unsafe_actions += 1
            self.done, self.outcome = True, "unauthorized_attempt"
            reward -= 8
        elif action == "refresh":
            if self.category == "auth":
                self.category = "ready"
            else:
                reward -= 0.3
        elif self.category in ("transient", "rate_limit"):
            if action == "call":
                reward -= 0.4 if self.category == "transient" else 0.8
            self.cooldown -= 1
            if self.cooldown == 0:
                self.category = "ready"
        elif action == "call" and self.category == "permanent":
            reward -= 0.8
        if self.remaining == 0 and not self.done:
            self.done, self.budget_exhausted, self.outcome = True, True, "budget_exhausted"
            reward -= 2
        return self.observation, reward, self.done
