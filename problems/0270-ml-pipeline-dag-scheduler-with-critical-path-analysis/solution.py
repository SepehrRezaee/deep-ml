import heapq
from collections import defaultdict


_EMPTY_ANALYSIS = {
    "execution_order": [],
    "earliest_start": {},
    "earliest_finish": {},
    "latest_start": {},
    "latest_finish": {},
    "slack": {},
    "critical_path": [],
    "makespan": 0,
}


def analyze_ml_pipeline(tasks: list) -> dict:
    """Analyze scheduling and critical-path timing for an ML pipeline DAG."""
    if not tasks:
        return _empty_result()

    durations = {task["id"]: task["duration"] for task in tasks}
    dependencies = {
        task["id"]: list(task.get("dependencies", []))
        for task in tasks
    }

    successors, indegrees = _build_dependency_graph(
        task_ids=durations,
        dependencies=dependencies,
    )
    execution_order = _topological_order(
        task_ids=durations,
        successors=successors,
        indegrees=indegrees,
    )

    earliest_start, earliest_finish = _forward_pass(
        execution_order=execution_order,
        durations=durations,
        dependencies=dependencies,
    )
    makespan = max(earliest_finish.values(), default=0)

    latest_start, latest_finish = _backward_pass(
        execution_order=execution_order,
        durations=durations,
        successors=successors,
        makespan=makespan,
    )

    slack = {
        task_id: latest_start[task_id] - earliest_start[task_id]
        for task_id in execution_order
    }
    critical_path = [
        task_id
        for task_id in execution_order
        if abs(slack[task_id]) < 1e-9
    ]

    return {
        "execution_order": execution_order,
        "earliest_start": earliest_start,
        "earliest_finish": earliest_finish,
        "latest_start": latest_start,
        "latest_finish": latest_finish,
        "slack": slack,
        "critical_path": critical_path,
        "makespan": makespan,
    }


def _build_dependency_graph(task_ids, dependencies):
    successors = defaultdict(list)
    indegrees = {task_id: 0 for task_id in task_ids}

    for task_id in task_ids:
        for dependency_id in dependencies[task_id]:
            successors[dependency_id].append(task_id)
            indegrees[task_id] += 1

    return successors, indegrees


def _topological_order(task_ids, successors, indegrees):
    ready = [
        task_id
        for task_id in task_ids
        if indegrees[task_id] == 0
    ]
    heapq.heapify(ready)

    order = []

    while ready:
        task_id = heapq.heappop(ready)
        order.append(task_id)

        for successor_id in successors[task_id]:
            indegrees[successor_id] -= 1

            if indegrees[successor_id] == 0:
                heapq.heappush(ready, successor_id)

    return order


def _forward_pass(execution_order, durations, dependencies):
    earliest_start = {}
    earliest_finish = {}

    for task_id in execution_order:
        start = max(
            (
                earliest_finish[dependency_id]
                for dependency_id in dependencies[task_id]
            ),
            default=0,
        )
        earliest_start[task_id] = start
        earliest_finish[task_id] = start + durations[task_id]

    return earliest_start, earliest_finish


def _backward_pass(execution_order, durations, successors, makespan):
    latest_start = {}
    latest_finish = {}

    for task_id in reversed(execution_order):
        finish = min(
            (
                latest_start[successor_id]
                for successor_id in successors[task_id]
            ),
            default=makespan,
        )
        latest_finish[task_id] = finish
        latest_start[task_id] = finish - durations[task_id]

    return (
        {task_id: latest_start[task_id] for task_id in execution_order},
        {task_id: latest_finish[task_id] for task_id in execution_order},
    )


def _empty_result():
    return {
        key: value.copy() if isinstance(value, (dict, list)) else value
        for key, value in _EMPTY_ANALYSIS.items()
    }