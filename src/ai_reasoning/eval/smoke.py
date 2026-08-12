from __future__ import annotations

import argparse
import time

from ai_reasoning.oracle.constraints import solve_constraints, validate_ordering
from ai_reasoning.oracle.graphs import solve_graph, validate_next_step
from ai_reasoning.tasks.constraints import generate_constraint_task
from ai_reasoning.tasks.graphs import generate_graph_task


def run(episodes: int) -> dict[str, float | int]:
    if episodes < 2:
        raise ValueError("episodes must be at least 2")

    constraints_n = episodes // 2
    graphs_n = episodes - constraints_n
    started = time.perf_counter()

    for seed in range(constraints_n):
        task = generate_constraint_task(seed=seed, entity_count=6)
        result = solve_constraints(task, max_solutions=4)
        if not validate_ordering(task, result.answer):
            raise AssertionError(f"invalid constraint oracle answer for seed {seed}")

    for seed in range(graphs_n):
        task = generate_graph_task(seed=seed, entity_count=8)
        result = solve_graph(task)
        if not validate_next_step(task, result.answer):
            raise AssertionError(f"invalid graph oracle answer for seed {seed}")

    elapsed = time.perf_counter() - started
    return {
        "episodes": episodes,
        "constraint_episodes": constraints_n,
        "graph_episodes": graphs_n,
        "elapsed_seconds": elapsed,
        "episodes_per_second": episodes / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10_000)
    args = parser.parse_args()
    result = run(args.episodes)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
