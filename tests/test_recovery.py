import random

import pytest

from tool_policy_lab.environment import Scenario, ToolEnvironment, draw_scenario
from tool_policy_lab.evaluation import evaluate_models, held_out_scenarios, play
from tool_policy_lab.training import greedy, load_checkpoint, save_checkpoint, train


def test_auth_attempt_is_rejected_and_terminal_state_cannot_mutate():
    env = ToolEnvironment(Scenario(("auth",), (1,)))
    _, reward, done = env.step("call")
    assert done and reward < 0 and env.unsafe_actions == 1
    assert env.completed == 0
    before = env.observation
    with pytest.raises(RuntimeError):
        env.step("call")
    assert env.observation == before


def test_recovery_sequence_respects_wait_refresh_and_completed_work():
    env = ToolEnvironment(Scenario(("rate_limit", "auth"), (2, 1)))
    env.step("wait")
    assert env.category == "rate_limit"
    env.step("wait")
    assert env.category == "ready"
    env.step("call")
    assert env.completed == 1 and env.category == "auth"
    env.step("refresh")
    assert env.category == "ready"
    env.step("call")
    assert env.outcome == "completed" and env.completed == 2


def test_budget_cannot_be_overrun_and_exhaustion_is_not_success():
    scenario = Scenario(("transient", "ready"), (2, 1), budget=2)
    result = play(scenario, lambda observation: "wait", "wait-forever")
    assert result["outcome"] == "budget_exhausted"
    assert not result["success"]
    assert len(result["steps"]) == 2
    assert result["steps"][-1]["remaining_budget"] == 0


def test_abort_is_not_counted_as_work_even_if_cost_is_low():
    scenario = Scenario(("ready", "ready", "ready"), (1, 1, 1))
    abort = play(scenario, lambda observation: "abort", "abort-all")
    assert not abort["success"] and abort["outcome"] == "aborted"
    assert abort["steps"][0]["completed"] == 0


def test_unknown_action_does_not_consume_budget():
    env = ToolEnvironment(Scenario(("ready",), (1,)))
    with pytest.raises(ValueError):
        env.step("make-up-action")
    assert env.remaining == env.scenario.budget


def test_parameterized_sequences_are_partitioned_independent_of_random_seed():
    evaluation = held_out_scenarios(160)
    evaluation_ids = {scenario.identity for scenario in evaluation}
    assert len(evaluation_ids) == 160
    assert all(scenario.partition == "evaluation" for scenario in evaluation)
    rng = random.Random(11)
    training = [draw_scenario(rng, "training") for _ in range(2000)]
    assert not evaluation_ids.intersection(scenario.identity for scenario in training)
    assert all(scenario.partition == "training" for scenario in training)


def test_train_is_reproducible_and_never_consumes_evaluation_scenarios():
    first, second = train(11, 500), train(11, 500)
    assert first == second
    assert any(value != 0 for values in first["q"].values() for value in values)
    assert not set(first["training_scenario_ids"]).intersection(scenario.identity for scenario in held_out_scenarios(160))


def test_checkpoint_round_trip_preserves_greedy_behavior(tmp_path):
    result = train(23, 500)
    path = tmp_path / "policy.json"
    save_checkpoint(result, path)
    loaded = load_checkpoint(path)
    assert loaded == result["q"]
    scenario = held_out_scenarios(1)[0]
    original = play(scenario, lambda observation: greedy(result["q"], observation), "original")
    restored = play(scenario, lambda observation: greedy(loaded, observation), "original")
    assert restored == original


def test_learned_checkpoint_beats_random_on_unseen_sequences_without_assuming_heuristic_win():
    model = train(11, 4000)
    comparisons, runs = evaluate_models([model], held_out_scenarios(120))
    learned, heuristic, fixed, random_result = comparisons
    assert learned["success_rate"] > random_result["success_rate"]
    assert learned["success_rate"] > fixed["success_rate"]
    assert heuristic["success_rate"] >= learned["success_rate"]
    assert all(len(run["steps"]) <= 14 for run in runs)
    assert learned["episodes"] == 120


def test_oversized_evaluation_request_fails_instead_of_drawing_forever():
    with pytest.raises(ValueError):
        held_out_scenarios(1001)


def test_invalid_scenarios_fail_before_execution():
    with pytest.raises(ValueError):
        Scenario(("made-up",), (1,))
    with pytest.raises(ValueError):
        Scenario(("ready",), (1,), budget=0)
