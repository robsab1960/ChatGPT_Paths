# Version 50
#
# Adds safe guard-rails for path explosion:
#   --max-seconds  Stop DFS after N seconds
#   --max-calls    Stop DFS after N recursive calls
#   --max-paths    Stop DFS after N paths found
# All limits are optional. When a limit is hit, the program exits gracefully with a clear message.

import sys
import re
import time
import argparse
from collections import defaultdict


class SearchLimitReached(Exception):
    """Raised when DFS exceeds user-provided limits (time/calls/paths)."""
    pass


class ControlFlowGraph:
    def __init__(self, verbose: bool = False):
        # Initialize the graph as a dictionary of adjacency lists
        self.graph = defaultdict(list)
        self.edges = set()  # Store unique edges
        self.nodes = set()  # Store unique nodes
        self.paths = []  # List to store all found paths

        # Verbose tracing controls
        self.verbose = verbose
        self.dfs_calls = 0  # counts recursive DFS calls (useful for diagnosing path explosion)

        # Optional DFS limits (None means "no limit")
        self.max_seconds = None
        self.max_calls = None
        self.max_paths = None

        # Internal timer start for max_seconds
        self._dfs_t0 = None

    def add_edge(self, u, v):
        # Ensure node names are stripped of extra whitespace
        u, v = u.strip(), v.strip()

        # Detect control characters in node names
        if any(ord(c) < 32 or ord(c) == 127 for c in u + v):
            print(f"Error: Invalid node name detected: '{u}' or '{v}' contains control characters.")
            sys.exit(1)

        # Validate node names (must be alphanumeric, underscores, or hyphens)
        if not re.match(r'^[\w-]+$', u) or not re.match(r'^[\w-]+$', v):
            print(f"Error: Invalid node name: '{u}' or '{v}' (contains spaces or special characters)")
            sys.exit(1)

        # Add a directed edge from node u to node v
        self.graph[u].append(v)
        self.edges.add((u, v))  # Store edge for complexity calculation
        self.nodes.add(u)
        self.nodes.add(v)

    def find_basis_set(self, start, end):
        # Perform depth-first search (DFS) to find all paths from start to end
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
            print(f"[INFO] Starting DFS path enumeration ({lims_str})...")

        self._dfs(start, end, path)

        if self.verbose:
            print(f"[INFO] DFS complete: calls={self.dfs_calls}, total_paths={len(self.paths)}")

        if not self.paths:
            print("Error: No paths found from start to end.")
            sys.exit(1)

        if self.verbose:
            print("[INFO] Extracting basis set...")

        basis = self._extract_basis_set()

        if self.verbose:
            print(f"[INFO] Basis extraction complete: basis_paths={len(basis)}")

        return basis

    def _check_limits(self):
        """Fast limit checks to prevent unbounded path explosion."""
        if self.max_calls is not None and self.dfs_calls > self.max_calls:
            raise SearchLimitReached(f"DFS call limit exceeded ({self.max_calls}).")

        if self.max_paths is not None and len(self.paths) >= self.max_paths:
            raise SearchLimitReached(f"Path limit exceeded ({self.max_paths}).")

        if self.max_seconds is not None:
            elapsed = time.perf_counter() - self._dfs_t0
            if elapsed > self.max_seconds:
                raise SearchLimitReached(f"Time limit exceeded ({self.max_seconds} seconds).")

    def _dfs(self, node, end, path):
        # Count DFS calls to help diagnose explosive growth
        self.dfs_calls += 1

        # Enforce limits (if configured)
        self._check_limits()

        if self.verbose and self.dfs_calls % 10000 == 0:
            print(f"[DFS] calls={self.dfs_calls}, paths_found={len(self.paths)}, depth={len(path)}")

        path.append(node)  # Add current node to path

        if node == end:
            self.paths.append(list(path))  # Store a copy of the path

            # Printing every found path is expensive; print a lightweight marker occasionally.
            if self.verbose and (len(self.paths) <= 10 or len(self.paths) % 1000 == 0):
                print(f"[DFS] path found: total_paths={len(self.paths)}, last_path_len={len(path)}")
        else:
            for neighbor in self.graph[node]:
                if path.count(neighbor) < 2:  # Allow revisiting nodes, but limit cycles
                    self._dfs(neighbor, end, path)

        path.pop()  # Remove the last node before backtracking

    def _extract_basis_set(self):
        # Compute cyclomatic complexity with correct node counting
        num_nodes = len(self.nodes)
        num_edges = len(self.edges)
        expected_basis_size = num_edges - num_nodes + 2

        # Ensure we collect the correct number of independent paths
        basis_set = []
        covered_edges = set()

        for path in self.paths:
            path_edges = {(path[i], path[i + 1]) for i in range(len(path) - 1)}

            # Always add paths that introduce new edges
            if not path_edges.issubset(covered_edges):
                basis_set.append(path)
                covered_edges.update(path_edges)

            # Ensure we collect enough paths to match cyclomatic complexity
            if len(basis_set) >= expected_basis_size:
                break

        # If not enough paths have been found, keep adding remaining unique paths
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

    @staticmethod
    def from_file(filename, verbose: bool = False):
        # Check if file exists
        try:
            with open(filename, 'r') as file:
                lines = file.read().splitlines()
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            sys.exit(1)

        # Check if file is empty or contains only whitespace before processing
        if not lines or all(line.strip() == "" for line in lines):
            print("Error: Input file is empty or contains only whitespace.")
            sys.exit(1)

        # Trim trailing blank or empty lines
        while lines and lines[-1].strip() == "":
            lines.pop()

        # Check if the file has at least one edge defined before processing
        if len(lines[1:-1]) == 0:
            print("Error: No edges defined in the input file.")
            sys.exit(1)

        # Check for edges with too few or too many nodes before processing
        for line in lines[1:-1]:
            parts = line.split()
            if len(parts) < 2:
                print(f"Error: Edge definition '{line}' is incomplete. Expected format: 'NODE1 NODE2'")
                sys.exit(1)
            if len(parts) > 2:
                print(f"Error: Edge definition '{line}' has too many nodes. Expected format: 'NODE1 NODE2'")
                sys.exit(1)

        # Check for duplicate edges before processing
        seen_edges = set()
        for line in lines[1:-1]:
            parts = line.split()
            u, v = map(str.strip, parts)
            if (u, v) in seen_edges:
                print(f"Error: Duplicate edge detected: '{u} -> {v}'")
                sys.exit(1)
            seen_edges.add((u, v))

        # Validate file format
        if len(lines) < 3:
            print("Error: Invalid input file format. Expected at least a start node, edges, and an end node.")
            sys.exit(1)

        start = lines[0].strip()   # First line is the start node
        end = lines[-1].strip()    # Last line is the end node
        cfg = ControlFlowGraph(verbose=verbose)

        # Check that no edges start at END and no edges end at START
        for line in lines[1:-1]:
            parts = line.split()
            u, v = map(str.strip, parts)
            if u == end:
                print("Error: end node should not start an edge")
                sys.exit(1)
            if v == start:
                print("Error: start node should not end an edge")
                sys.exit(1)

        for line in lines[1:-1]:
            parts = line.split()
            u, v = map(str.strip, parts)
            cfg.add_edge(u, v)

        # Check if start and end nodes are in the edges
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
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Stop DFS after this many seconds (prevents path explosion)")
    parser.add_argument("--max-calls", type=int, default=None,
                        help="Stop DFS after this many recursive calls (prevents path explosion)")
    parser.add_argument("--max-paths", type=int, default=None,
                        help="Stop DFS after finding this many paths (prevents path explosion)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time.perf_counter() if args.time else None

    cfg = None  # so we can report partial progress on limit-hit

    try:
        cfg, start, end = ControlFlowGraph.from_file(args.filename, verbose=args.verbose)

        # Apply optional DFS limits
        cfg.max_seconds = args.max_seconds
        cfg.max_calls = args.max_calls
        cfg.max_paths = args.max_paths

        basis_set = cfg.find_basis_set(start, end)

        print("Basis Set of Paths:")
        for path in basis_set:
            print(" -> ".join(path))

        print(f"Number of Nodes: {cfg.num_nodes}")
        print(f"Number of Edges: {cfg.num_edges}")
        print(f"Cyclomatic Complexity (Expected Basis Paths): {cfg.cyclomatic_complexity}")
        print(f"Number of Basis Paths Found: {cfg.basis_path_count}")

    except SearchLimitReached as e:
        # Graceful termination when a user-provided limit is hit
        print(f"Stopped early due to search limits: {e}")
        calls = getattr(cfg, "dfs_calls", 0) if cfg is not None else 0
        paths_found = len(getattr(cfg, "paths", [])) if cfg is not None else 0
        print(f"Progress so far: calls={calls}, paths_found={paths_found}")
        sys.exit(1)

    finally:
        if args.time and t0 is not None:
            elapsed = time.perf_counter() - t0
            print(f"Elapsed Time (seconds): {elapsed:.6f}")
