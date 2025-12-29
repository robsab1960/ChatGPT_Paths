# Version 51
#
# Adds --linear option to compute a basis set of paths without enumerating all paths.
#   Default (no flag): enumeration-based DFS (can be exponential; guarded by --max-* limits).
#   --linear: O(E+V) spanning-tree + chord construction (scales to large graphs).

import sys
import re
import time
import argparse
from collections import defaultdict, deque


class SearchLimitReached(Exception):
    """Raised when DFS exceeds user-provided limits (time/calls/paths)."""
    pass


class ControlFlowGraph:
    def __init__(self, verbose: bool = False):
        # Graph representation
        self.graph = defaultdict(list)  # adjacency list (directed)
        self.edges = set()              # unique directed edges
        self.nodes = set()              # unique nodes

        # Path enumeration outputs (enumeration mode only)
        self.paths = []                 # all found START->END paths (can be huge)

        # Verbose tracing controls
        self.verbose = verbose
        self.dfs_calls = 0              # counts recursive DFS calls

        # Optional DFS limits (None means "no limit") for enumeration mode
        self.max_seconds = None
        self.max_calls = None
        self.max_paths = None
        self._dfs_t0 = None             # internal timer start for max_seconds

        # Reporting fields (set by whichever algorithm runs)
        self.num_nodes = 0
        self.num_edges = 0
        self.basis_path_count = 0
        self.cyclomatic_complexity = 0

    def add_edge(self, u, v):
        # Ensure node names are stripped of extra whitespace
        u, v = u.strip(), v.strip()

        # Detect control characters in node names
        if any(ord(c) < 32 or ord(c) == 127 for c in (u + v)):
            print(f"Error: Invalid node name detected: '{u}' or '{v}' contains control characters.")
            sys.exit(1)

        # Validate node names (must be alphanumeric, underscores, or hyphens)
        if not re.match(r'^[\w-]+$', u) or not re.match(r'^[\w-]+$', v):
            print(f"Error: Invalid node name: '{u}' or '{v}' (contains spaces or special characters)")
            sys.exit(1)

        # Add a directed edge from node u to node v
        self.graph[u].append(v)
        self.edges.add((u, v))
        self.nodes.add(u)
        self.nodes.add(v)

    # ---------------------------------------------------------------------
    # ENUMERATION MODE (existing behavior; can be exponential)
    # ---------------------------------------------------------------------
    def find_basis_set(self, start, end):
        """Enumeration-based algorithm: enumerate bounded paths, then select a basis."""
        path = []
        self.paths.clear()
        self.dfs_calls = 0
        self._dfs_t0 = time.perf_counter()

        if self.verbose:
            lims = []
            if self.max_seconds is not None:
                lims.append(f"max_seconds={self.max_seconds}")
            if self.max_calls is not None:
                lims.append(f"max_calls={self.max_calls}")
            if self.max_paths is not None:
                lims.append(f"max_paths={self.max_paths}")
            lims_str = (", ".join(lims)) if lims else "no limits"
            print("[INFO] Using enumeration algorithm")
            print(f"[INFO] Starting DFS path enumeration ({lims_str})...")

        self._dfs(start, end, path)

        if self.verbose:
            print(f"[INFO] DFS complete: calls={self.dfs_calls}, total_paths={len(self.paths)}")

        if not self.paths:
            print("Error: No paths found from start to end.")
            sys.exit(1)

        if self.verbose:
            print("[INFO] Extracting basis set...")

        basis = self._extract_basis_set_from_enumerated_paths()

        if self.verbose:
            print(f"[INFO] Basis extraction complete: basis_paths={len(basis)}")

        return basis

    def _check_limits(self):
        if self.max_calls is not None and self.dfs_calls > self.max_calls:
            raise SearchLimitReached(f"DFS call limit exceeded ({self.max_calls}).")
        if self.max_paths is not None and len(self.paths) >= self.max_paths:
            raise SearchLimitReached(f"Path limit exceeded ({self.max_paths}).")
        if self.max_seconds is not None:
            elapsed = time.perf_counter() - self._dfs_t0
            if elapsed > self.max_seconds:
                raise SearchLimitReached(f"Time limit exceeded ({self.max_seconds} seconds).")

    def _dfs(self, node, end, path):
        self.dfs_calls += 1
        self._check_limits()

        if self.verbose and self.dfs_calls % 10000 == 0:
            print(f"[DFS] calls={self.dfs_calls}, paths_found={len(self.paths)}, depth={len(path)}")

        path.append(node)

        if node == end:
            self.paths.append(list(path))
            if self.verbose and (len(self.paths) <= 10 or len(self.paths) % 1000 == 0):
                print(f"[DFS] path found: total_paths={len(self.paths)}, last_path_len={len(path)}")
        else:
            for neighbor in self.graph[node]:
                # Allow revisiting nodes, but limit cycles
                if path.count(neighbor) < 2:
                    self._dfs(neighbor, end, path)

        path.pop()

    def _extract_basis_set_from_enumerated_paths(self):
        # Cyclomatic complexity computed on the full graph that was parsed
        num_nodes = len(self.nodes)
        num_edges = len(self.edges)
        expected_basis_size = num_edges - num_nodes + 2

        basis_set = []
        covered_edges = set()

        for path in self.paths:
            path_edges = {(path[i], path[i + 1]) for i in range(len(path) - 1)}
            if not path_edges.issubset(covered_edges):
                basis_set.append(path)
                covered_edges.update(path_edges)
            if len(basis_set) >= expected_basis_size:
                break

        if len(basis_set) < expected_basis_size:
            remaining_paths = [p for p in self.paths if p not in basis_set]
            for path in remaining_paths:
                basis_set.append(path)
                if len(basis_set) >= expected_basis_size:
                    break

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.basis_path_count = len(basis_set)
        self.cyclomatic_complexity = expected_basis_size
        return basis_set

    # ---------------------------------------------------------------------
    # LINEAR MODE (O(E+V)): spanning-tree + chord construction
    # ---------------------------------------------------------------------
    def _reachable_from(self, start):
        seen = set()
        stack = [start]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            for v in self.graph[u]:
                if v not in seen:
                    stack.append(v)
        return seen

    def _reverse_adjacency(self):
        rev = defaultdict(list)
        for (u, v) in self.edges:
            rev[v].append(u)
        return rev

    def _reachable_to_end(self, end, rev):
        seen = set()
        stack = [end]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            for p in rev[u]:
                if p not in seen:
                    stack.append(p)
        return seen

    def _build_forward_tree_parents(self, start, allowed):
        """BFS tree from start on allowed subgraph; parent[v]=u."""
        parent = {start: None}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self.graph[u]:
                if v in allowed and v not in parent:
                    parent[v] = u
                    q.append(v)
        return parent

    def _build_reverse_tree_next_hop(self, end, allowed, rev):
        """BFS from end on reversed edges; next_hop[x] is next node on x->...->end."""
        next_hop = {end: None}
        q = deque([end])
        while q:
            u = q.popleft()
            for p in rev[u]:  # p -> u in original
                if p in allowed and p not in next_hop:
                    next_hop[p] = u
                    q.append(p)
        return next_hop

    def _path_start_to(self, parent, u):
        if u not in parent:
            return None
        out = []
        cur = u
        while cur is not None:
            out.append(cur)
            cur = parent[cur]
        out.reverse()
        return out

    def _path_to_end(self, next_hop, u):
        if u not in next_hop:
            return None
        out = [u]
        cur = u
        while next_hop[cur] is not None:
            cur = next_hop[cur]
            out.append(cur)
        return out

    def find_basis_set_linear(self, start, end):
        """Linear-time basis path construction (no path enumeration)."""
        if self.verbose:
            print("[INFO] Using linear (spanning-tree) algorithm")

        # Restrict to nodes that lie on at least one start->end route:
        # allowed = reachable_from_start ∩ can_reach_end
        reach = self._reachable_from(start)
        rev = self._reverse_adjacency()
        coreach = self._reachable_to_end(end, rev)
        allowed = reach & coreach

        if start not in allowed or end not in allowed:
            print("Error: No paths found from start to end.")
            sys.exit(1)

        # Build trees
        parent = self._build_forward_tree_parents(start, allowed)
        next_hop = self._build_reverse_tree_next_hop(end, allowed, rev)

        base = self._path_start_to(parent, end)
        if base is None:
            print("Error: No paths found from start to end.")
            sys.exit(1)

        # Tree edges in forward tree
        tree_edges = set()
        for node, p in parent.items():
            if p is not None:
                tree_edges.add((p, node))

        # Allowed edges (unique)
        allowed_edges = {(u, v) for (u, v) in self.edges if u in allowed and v in allowed}

        Vp = len(allowed)
        Ep = len(allowed_edges)
        expected = Ep - Vp + 2  # cyclomatic complexity on the relevant subgraph

        if self.verbose:
            print(f"[INFO] Relevant subgraph: nodes={Vp}, edges={Ep}, expected_basis={expected}")

        # Construct basis paths: base path + one per chord (non-tree edge)
        basis = [base]
        seen_paths = {tuple(base)}

        # Iterate chords in stable order for reproducibility
        for (u, v) in sorted(allowed_edges):
            if (u, v) in tree_edges:
                continue  # tree edge -> not a chord

            p1 = self._path_start_to(parent, u)   # START -> u
            p2 = self._path_to_end(next_hop, v)   # v -> END

            if p1 is None or p2 is None:
                continue

            candidate = p1 + [v] + p2[1:]  # ...u, then chord to v, then v..END (skip duplicate v)
            t = tuple(candidate)
            if t not in seen_paths:
                seen_paths.add(t)
                basis.append(candidate)

            if len(basis) >= expected:
                break

        # Store reporting values (for the relevant subgraph)
        self.num_nodes = Vp
        self.num_edges = Ep
        self.cyclomatic_complexity = expected
        self.basis_path_count = len(basis)

        if len(basis) != expected and self.verbose:
            print(f"[WARN] Constructed basis size {len(basis)} != expected {expected}.")

        return basis

    # ---------------------------------------------------------------------
    # FILE PARSING (unchanged validation)
    # ---------------------------------------------------------------------
    @staticmethod
    def from_file(filename, verbose: bool = False):
        try:
            with open(filename, 'r') as file:
                lines = file.read().splitlines()
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            sys.exit(1)

        if not lines or all(line.strip() == "" for line in lines):
            print("Error: Input file is empty or contains only whitespace.")
            sys.exit(1)

        # Trim trailing blank lines
        while lines and lines[-1].strip() == "":
            lines.pop()

        if len(lines[1:-1]) == 0:
            print("Error: No edges defined in the input file.")
            sys.exit(1)

        # Edge token count checks
        for line in lines[1:-1]:
            parts = line.split()
            if len(parts) < 2:
                print(f"Error: Edge definition '{line}' is incomplete. Expected format: 'NODE1 NODE2'")
                sys.exit(1)
            if len(parts) > 2:
                print(f"Error: Edge definition '{line}' has too many nodes. Expected format: 'NODE1 NODE2'")
                sys.exit(1)

        # Duplicate edge check
        seen_edges = set()
        for line in lines[1:-1]:
            u, v = map(str.strip, line.split())
            if (u, v) in seen_edges:
                print(f"Error: Duplicate edge detected: '{u} -> {v}'")
                sys.exit(1)
            seen_edges.add((u, v))

        if len(lines) < 3:
            print("Error: Invalid input file format. Expected at least a start node, edges, and an end node.")
            sys.exit(1)

        start = lines[0].strip()
        end = lines[-1].strip()
        cfg = ControlFlowGraph(verbose=verbose)

        # START/END loop guards:
        # - no edges start at END
        # - no edges end at START
        for line in lines[1:-1]:
            u, v = map(str.strip, line.split())
            if u == end:
                print("Error: end node should not start an edge")
                sys.exit(1)
            if v == start:
                print("Error: start node should not end an edge")
                sys.exit(1)

        for line in lines[1:-1]:
            u, v = map(str.strip, line.split())
            cfg.add_edge(u, v)

        if start not in cfg.nodes:
            print(f"Error: STARTING NODE '{start}' is missing.")
            sys.exit(1)
        if end not in cfg.nodes:
            print(f"Error: ENDING NODE '{end}' is missing.")
            sys.exit(1)

        if verbose:
            print(f"[INFO] Parsed CFG: nodes={len(cfg.nodes)}, edges={len(cfg.edges)}, start={start}, end={end}")

        return cfg, start, end


def parse_args():
    parser = argparse.ArgumentParser(description="Compute a basis set of paths from a CFG file.")
    parser.add_argument("filename", help="Input CFG text file")
    parser.add_argument("-t", "--time", action="store_true", help="Print elapsed execution time")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose tracing while running")
    parser.add_argument("--linear", action="store_true",
                        help="Use O(E+V) spanning-tree algorithm instead of path enumeration")

    # Enumeration-mode guard rails (still accepted even if --linear is used; they just won't apply)
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Stop DFS after this many seconds (enumeration mode only)")
    parser.add_argument("--max-calls", type=int, default=None,
                        help="Stop DFS after this many recursive calls (enumeration mode only)")
    parser.add_argument("--max-paths", type=int, default=None,
                        help="Stop DFS after finding this many paths (enumeration mode only)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time.perf_counter() if args.time else None

    cfg = None  # so we can report partial progress on limit-hit

    try:
        cfg, start, end = ControlFlowGraph.from_file(args.filename, verbose=args.verbose)

        # Apply optional DFS limits (enumeration mode)
        cfg.max_seconds = args.max_seconds
        cfg.max_calls = args.max_calls
        cfg.max_paths = args.max_paths

        if args.linear:
            basis_set = cfg.find_basis_set_linear(start, end)
        else:
            basis_set = cfg.find_basis_set(start, end)

        print("Basis Set of Paths:")
        for path in basis_set:
            print(" -> ".join(path))

        print(f"Number of Nodes: {cfg.num_nodes}")
        print(f"Number of Edges: {cfg.num_edges}")
        print(f"Cyclomatic Complexity (Expected Basis Paths): {cfg.cyclomatic_complexity}")
        print(f"Number of Basis Paths Found: {cfg.basis_path_count}")

    except SearchLimitReached as e:
        print(f"Stopped early due to search limits: {e}")
        calls = getattr(cfg, "dfs_calls", 0) if cfg is not None else 0
        paths_found = len(getattr(cfg, "paths", [])) if cfg is not None else 0
        print(f"Progress so far: calls={calls}, paths_found={paths_found}")
        sys.exit(1)

    finally:
        if args.time and t0 is not None:
            elapsed = time.perf_counter() - t0
            print(f"Elapsed Time (seconds): {elapsed:.6f}")
