import random

from ai_reasoning.oracle.constraints import solve_constraints, validate_ordering
from ai_reasoning.oracle.graphs import solve_graph, validate_next_step
from ai_reasoning.tasks import generate_constraint_task, generate_graph_task, rename_entities


def test_constraint_generation_is_deterministic():
    assert generate_constraint_task(42, 7) == generate_constraint_task(42, 7)


def test_graph_generation_is_deterministic():
    assert generate_graph_task(42, 10) == generate_graph_task(42, 10)


def test_generated_constraint_tasks_are_satisfiable():
    for seed in range(100):
        task = generate_constraint_task(seed, entity_count=6)
        result = solve_constraints(task, max_solutions=8)
        assert validate_ordering(task, result.answer)


def test_generated_graph_tasks_have_valid_shortest_next_step():
    for seed in range(100):
        task = generate_graph_task(seed, entity_count=8)
        result = solve_graph(task)
        assert validate_next_step(task, result.answer)
        assert result.path_length >= 1


def test_constraint_renaming_preserves_solution():
    task = generate_constraint_task(901, entity_count=7)
    solution = solve_constraints(task, max_solutions=1).answer
    mapping = list(range(task.entity_count))
    random.Random(77).shuffle(mapping)
    mapping = tuple(mapping)

    renamed = rename_entities(task, mapping)
    renamed_solution = tuple(mapping[node] for node in solution)
    assert validate_ordering(renamed, renamed_solution)
    assert renamed.stats == task.stats


def test_graph_renaming_preserves_shortest_path_length():
    task = generate_graph_task(812, entity_count=10, forbidden_count=2)
    original = solve_graph(task)
    mapping = list(range(task.entity_count))
    random.Random(88).shuffle(mapping)
    mapping = tuple(mapping)

    renamed = rename_entities(task, mapping)
    renamed_result = solve_graph(renamed)
    assert renamed_result.path_length == original.path_length
    assert mapping[original.answer] in renamed_result.valid_answers
    assert renamed.stats == task.stats
