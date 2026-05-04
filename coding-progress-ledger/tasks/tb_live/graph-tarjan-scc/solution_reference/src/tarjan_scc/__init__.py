def scc(adj: dict[int, list[int]]) -> list[list[int]]:
    nodes = set(adj)
    for neighbors in adj.values():
        nodes.update(neighbors)
    if not nodes:
        return []

    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    on_stack = {}
    components = []

    def strongconnect(v):
        index[v] = lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in adj.get(v, []):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w):
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            components.append(sorted(component))

    for v in nodes:
        if v not in index:
            strongconnect(v)

    return sorted(components, key=lambda c: c[0])
