import heapq
from collections import defaultdict


def analyze_ml_pipeline(tasks: list) -> dict:
    """
    Analyze an ML pipeline DAG for scheduling and critical path.

    Args:
        tasks: list of task dicts with:
            - 'id': task identifier (str)
            - 'duration': task duration in minutes (int)
            - 'dependencies': list of task IDs this task depends on

    Returns:
        dict with:
            - 'execution_order': topologically sorted list of task IDs
            - 'earliest_start': dict mapping task ID to earliest start time
            - 'earliest_finish': dict mapping task ID to earliest finish time
            - 'latest_start': dict mapping task ID to latest start time
            - 'latest_finish': dict mapping task ID to latest finish time
            - 'slack': dict mapping task ID to slack time
            - 'critical_path': list of task IDs on critical path (in execution order)
            - 'makespan': total time to complete pipeline
    """
    result = {
        'execution_order': [],
        'earliest_start': {},
        'earliest_finish': {},
        'latest_start': {},
        'latest_finish': {},
        'slack': {},
        'critical_path': [],
        'makespan': 0,
    }
    if not tasks:
        return result

    durations = {task['id']: task['duration'] for task in tasks}
    dependencies = {
        task['id']: list(task.get('dependencies', []))
        for task in tasks
    }

    dependents = defaultdict(list)
    remaining_dependencies = {task_id: 0 for task_id in durations}
    for task_id, prerequisites in dependencies.items():
        for prerequisite in prerequisites:
            dependents[prerequisite].append(task_id)
            remaining_dependencies[task_id] += 1

    ready = [task_id for task_id, count in remaining_dependencies.items() if count == 0]
    heapq.heapify(ready)
    execution_order = []

    # The heap picks the alphabetically first task whenever several are ready.
    while ready:
        task_id = heapq.heappop(ready)
        execution_order.append(task_id)
        for dependent in dependents[task_id]:
            remaining_dependencies[dependent] -= 1
            if remaining_dependencies[dependent] == 0:
                heapq.heappush(ready, dependent)

    earliest_start = {}
    earliest_finish = {}
    for task_id in execution_order:
        start = max(
            (earliest_finish[prerequisite] for prerequisite in dependencies[task_id]),
            default=0,
        )
        earliest_start[task_id] = start
        earliest_finish[task_id] = start + durations[task_id]

    makespan = max(earliest_finish.values(), default=0)

    # Work backward to find how late each task can run without delaying the pipeline.
    latest_start = {}
    latest_finish = {}
    for task_id in reversed(execution_order):
        finish = min(
            (latest_start[dependent] for dependent in dependents[task_id]),
            default=makespan,
        )
        latest_finish[task_id] = finish
        latest_start[task_id] = finish - durations[task_id]

    slack = {
        task_id: latest_start[task_id] - earliest_start[task_id]
        for task_id in execution_order
    }
    critical_path = [
        task_id for task_id in execution_order
        if abs(slack[task_id]) < 1e-9
    ]

    result.update({
        'execution_order': execution_order,
        'earliest_start': earliest_start,
        'earliest_finish': earliest_finish,
        'latest_start': {task_id: latest_start[task_id] for task_id in execution_order},
        'latest_finish': {task_id: latest_finish[task_id] for task_id in execution_order},
        'slack': slack,
        'critical_path': critical_path,
        'makespan': makespan,
    })
    return result
