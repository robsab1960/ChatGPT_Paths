import sys
import random


def generate_graph(num_nodes: int, file_name: str, extra_edges: int = None, seed: int = None) -> None:
    """
    Generates a directed graph in the format:
      - first line: start node name
      - middle lines: edges "U V"
      - last line: end node name

    Guarantees:
      - at least one path from START to END
      - START is never the destination of an edge
      - END is never the source of an edge
      - no self-loops
      - no duplicate edges
    """

    if seed is not None:
        random.seed(seed)

    # Create nodes Node1..NodeN
    nodes = [f"Node{i}" for i in range(1, num_nodes + 1)]
    start = nodes[0]
    end = nodes[-1]

    # Reasonable default edge density
    if extra_edges is None:
        extra_edges = 2 * (num_nodes - 1)

    edges = set()

    # --------------------------------------------------
    # 1) Backbone chain (guarantees START -> ... -> END)
    # --------------------------------------------------
    for i in range(num_nodes - 1):
        # Node1 -> Node2 -> ... -> NodeN
        edges.add((nodes[i], nodes[i + 1]))

    # --------------------------------------------------
    # 2) Add extra random edges
    # --------------------------------------------------
    max_attempts = extra_edges * 20
    attempts = 0

    while len(edges) < (num_nodes - 1) + extra_edges and attempts < max_attempts:
        attempts += 1

        u = random.choice(nodes)  # potential source
        v = random.choice(nodes)  # potential destination

        # Enforce CFG structural rules
        if u == v:
            continue                # no self-loops
        if v == start:
            continue                # START is never a destination
        if u == end:
            continue                # END is never a source

        edges.add((u, v))           # set prevents duplicates automatically

    # --------------------------------------------------
    # 3) Write to file in your required format
    # --------------------------------------------------
    with open(file_name, "w", newline="\n") as f:
        f.write(f"{start}\n")       # START node
        for u, v in sorted(edges):
            f.write(f"{u} {v}\n")
        f.write(f"{end}\n")         # END node


if __name__ == "__main__":
    # Usage:
    #   python script.py <num_nodes> <file_name> [extra_edges] [seed]

    if len(sys.argv) not in (3, 4, 5):
        print("Usage: python script.py <num_nodes> <file_name> [extra_edges] [seed]")
        sys.exit(1)

    try:
        num_nodes = int(sys.argv[1])
    except ValueError:
        print("Error: <num_nodes> must be a positive whole number.")
        sys.exit(1)

    file_name = sys.argv[2]

    if num_nodes < 2:
        print("Error: Number of nodes must be at least 2 (START and END).")
        sys.exit(1)

    extra_edges = None
    seed = None

    if len(sys.argv) >= 4:
        try:
            extra_edges = int(sys.argv[3])
            if extra_edges < 0:
                raise ValueError
        except ValueError:
            print("Error: [extra_edges] must be a non-negative whole number.")
            sys.exit(1)

    if len(sys.argv) == 5:
        try:
            seed = int(sys.argv[4])
        except ValueError:
            print("Error: [seed] must be a whole number.")
            sys.exit(1)

    generate_graph(num_nodes, file_name, extra_edges, seed)
