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

    duration = {t['id']: t['duration'] for t in tasks}
    deps = {t['id']: list(t.get('dependencies', [])) for t in tasks}

    succ = defaultdict(list)
    indeg = {tid: 0 for tid in duration}
    for tid in duration:
        for d in deps[tid]:
            succ[d].append(tid)
            indeg[tid] += 1

    # Kahn's algorithm with a min-heap for alphabetical tie-breaking
    heap = [tid for tid in duration if indeg[tid] == 0]
    heapq.heapify(heap)
    topo = []
    while heap:
        v = heapq.heappop(heap)
        topo.append(v)
        for u in succ[v]:
            indeg[u] -= 1
            if indeg[u] == 0:
                heapq.heappush(heap, u)

    # Forward pass: earliest start/finish
    es, ef = {}, {}
    for v in topo:
        es[v] = max((ef[d] for d in deps[v]), default=0)
        ef[v] = es[v] + duration[v]
    makespan = max(ef.values()) if ef else 0

    # Backward pass: latest start/finish
    ls, lf = {}, {}
    for v in reversed(topo):
        lf[v] = min((ls[u] for u in succ[v]), default=makespan)
        ls[v] = lf[v] - duration[v]

    # Emit all per-task dicts in execution (topological) order
    es = {v: es[v] for v in topo}
    ef = {v: ef[v] for v in topo}
    ls = {v: ls[v] for v in topo}
    lf = {v: lf[v] for v in topo}
    slack = {v: ls[v] - es[v] for v in topo}

    # Critical path: zero-slack tasks, in execution order
    critical_path = [v for v in topo if abs(slack[v]) < 1e-9]

    result.update({
        'execution_order': topo,
        'earliest_start': es,
        'earliest_finish': ef,
        'latest_start': ls,
        'latest_finish': lf,
        'slack': slack,
        'critical_path': critical_path,
        'makespan': makespan,
    })
    return result