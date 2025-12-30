# Version 63
#
# Improves --validate behavior when enumeration hits search limits.
# Adds --confirm-basis (strong rank-based validation of basis-ness).
#   - Always prints the LINEAR summary.
#   - If enumeration completes: prints both summaries and PASS/FAIL.
#   - If enumeration is stopped by --max-* limits: prints enumeration status + INCONCLUSIVE.
#
# Notes:
#   - Enumeration can still be expensive; use --max-seconds/--max-calls/--max-paths.
#   - Linear mode remains available via --linear for scalable runs.

import sys
import re
import time
import argparse
from collections import defaultdict, deque

import os
import socket
import platform
import hashlib
import datetime
import json


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_timestamp():
    # ISO 8601 UTC with microseconds
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _run_id():
    # timestamp::pid
    return f"{_utc_timestamp()}::pid{os.getpid()}"


def _relpath(p):
    try:
        return os.path.relpath(p).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def _analysis_dict(cfg, start, end, basis_paths, algorithm, confirm_enabled):
    allowed_nodes, allowed_edges = cfg.relevant_subgraph(start, end)
    nodes_list = sorted(allowed_nodes)
    edges_list = [[u, v] for (u, v) in sorted(allowed_edges)]
    Vp = len(nodes_list)
    Ep = len(edges_list)
    cyclo = Ep - Vp + 2

    analysis = {
        "algorithm": algorithm,
        "relevant_subgraph": {
            "nodes": nodes_list,
            "edges": edges_list,
            "node_count": Vp,
            "edge_count": Ep,
            "cyclomatic_complexity": cyclo
        },
        "basis": {
            "basis_count": len(basis_paths) if basis_paths is not None else 0,
            "paths": basis_paths if basis_paths is not None else []
        },
        "confirm_basis": {
            "enabled": bool(confirm_enabled),
            "rank_gf2": None,
            "pass": None,
            "failure_reason": None
        },
        "performance": {
            "elapsed_seconds": None
        }
    }

    if confirm_enabled and basis_paths is not None:
        # Compute rank without printing
        _, edge_to_index = build_edge_index({(u, v) for (u, v) in allowed_edges})
        rows = []
        ok_paths = True
        for p in basis_paths:
            m = path_to_incidence_mask(p, edge_to_index)
            if m is None:
                ok_paths = False
                break
            rows.append(m)
        rank = gf2_rank_bitmasks(rows) if ok_paths else 0
        analysis["confirm_basis"]["rank_gf2"] = rank
        analysis["confirm_basis"]["pass"] = (rank == cyclo and len(basis_paths) == cyclo and ok_paths)
        if not ok_paths:
            analysis["confirm_basis"]["failure_reason"] = "path contains edge not in relevant subgraph"
        elif len(basis_paths) != cyclo:
            analysis["confirm_basis"]["failure_reason"] = "basis size does not match cyclomatic complexity"
        elif rank != cyclo:
            analysis["confirm_basis"]["failure_reason"] = "paths are linearly dependent (rank != cyclomatic complexity)"
        else:
            analysis["confirm_basis"]["failure_reason"] = None

    return analysis


def make_json_payload(args, cfg, start, end, mode, analysis_obj, exit_code, status, message, elapsed_seconds):
    filename_abs = args.filename
    payload = {
        "schema_version": "1.1",
        "tool": {
            "name": "Path_Master",
            "version": _extract_version_number(),
            "source_file": _relpath(__file__),
            "git": {
                "commit": None,
                "dirty": None
            }
        },
        "run": {
            "run_id": _run_id(),
            "timestamp_utc": _utc_timestamp(),
            "host": socket.gethostname(),
            "platform": platform.platform()
        },
        "input": {
            "file_path": _relpath(filename_abs),
            "file_name": os.path.basename(filename_abs),
            "sha256": _sha256_file(filename_abs),
            "start_node": start,
            "end_node": end
        },
        "options": {
            "argv": sys.argv[1:],
            "mode": mode,
            "algorithm": ("both" if args.validate else ("linear" if args.linear else "enumeration")),
            "verbose": bool(args.verbose),
            "time_enabled": bool(args.time),
            "confirm_basis": bool(args.confirm_basis),
            "json_enabled": True
        },
        "limits": {
            "max_seconds": args.max_seconds,
            "max_calls": args.max_calls,
            "max_paths": args.max_paths
        },
        "results": {
            "summary": {
                "exit_code": exit_code,
                "status": status,
                "message": message
            },
            "analysis": analysis_obj
        },
        "diagnostics": {
            "stderr": None,
            "warnings": [],
            "enumeration_stats": {
                "dfs_calls": getattr(cfg, "dfs_calls", None),
                "paths_found": len(getattr(cfg, "paths", []) or []),
                "max_depth": getattr(cfg, "max_depth_seen", None)
            }
        }
    }

    # Set elapsed time in analysis objects
    if mode == "validate":
        if "linear" in payload["results"]["analysis"]:
            payload["results"]["analysis"]["linear"]["performance"]["elapsed_seconds"] = elapsed_seconds
        if "enumeration" in payload["results"]["analysis"]:
            payload["results"]["analysis"]["enumeration"]["performance"]["elapsed_seconds"] = elapsed_seconds
    else:
        payload["results"]["analysis"]["performance"]["elapsed_seconds"] = elapsed_seconds

    return payload


def _extract_version_number():
    # Reads the leading "# Version NN" line at top of file.
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        m = re.search(r"Version\s+(\d+)", first)
        return int(m.group(1)) if m else None
    except Exception:
        return None


HELP_EPILOG = """
PATH_MASTER(1)              User Commands              PATH_MASTER(1)

NAME
    Path_Master — compute a basis set of paths for a control flow graph

SYNOPSIS
    path_master.py [OPTIONS] FILE

DESCRIPTION
    Path_Master analyzes a control flow graph (CFG) and computes a basis set of
    execution paths suitable for McCabe basis path testing.

    Input format:
        - First line: START node
        - Last line:  END node
        - Intermediate lines: directed edges "NODE1 NODE2"

ALGORITHMS
    Default (no flag):
        Enumeration-based DFS (can be exponential; use --max-* limits)

    --linear:
        O(E + V) spanning-tree based algorithm (no path enumeration)

JSON OUTPUT
    --json
        Emit a single JSON object to stdout (schema_version 1.1). When enabled,
        human-readable output is suppressed.

VALIDATION
    --validate
        Runs both algorithms and checks invariants:
            - cyclomatic complexity on the START→END relevant subgraph
            - number of basis paths equals cyclomatic complexity
            - each basis path is a valid START→END path using existing edges

        If enumeration is stopped by --max-* limits, validation is INCONCLUSIVE.

CONFIRM-BASIS
    --confirm-basis
        Perform strong validation that the produced path set is a true basis.
        This converts basis paths into edge-incidence vectors (mod 2) and checks
        that their GF(2) rank equals cyclomatic complexity on the relevant subgraph.
        This does not enumerate all paths and is fast in practice.

ENUMERATION LIMITS (enumeration algorithm only)
    --max-seconds SECONDS
        Stop DFS after the given number of seconds.

    --max-calls COUNT
        Stop DFS after the given number of recursive calls.

    --max-paths COUNT
        Stop DFS after finding the given number of paths.

DIAGNOSTICS
    -v, --verbose
        Enable verbose tracing

    -t, --time
        Print elapsed execution time

EXIT STATUS
    0   Success / Validation PASS
    1   Error / Validation FAIL
    2   Validation INCONCLUSIVE (enumeration incomplete)

SEE ALSO
    McCabe (1976), NISTIR 5737, Cal Poly Basis Path Testing Tutorial
"""



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
        u, v = u.strip(), v.strip()

        # Detect control characters in node names
        if any(ord(c) < 32 or ord(c) == 127 for c in (u + v)):
            print(f"Error: Invalid node name detected: '{u}' or '{v}' contains control characters.")
            sys.exit(1)

        # Validate node names (must be alphanumeric, underscores, or hyphens)
        if not re.match(r'^[\w-]+$', u) or not re.match(r'^[\w-]+$', v):
            print(f"Error: Invalid node name: '{u}' or '{v}' (contains spaces or special characters)")
            sys.exit(1)

        self.graph[u].append(v)
        self.edges.add((u, v))
        self.nodes.add(u)
        self.nodes.add(v)

    # ---------------------------------------------------------------------
    # Common helpers
    # ---------------------------------------------------------------------
    def _reverse_adjacency(self):
        rev = defaultdict(list)
        for (u, v) in self.edges:
            rev[v].append(u)
        return rev

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

    def relevant_subgraph(self, start, end):
        """Return (allowed_nodes, allowed_edges_set) for nodes/edges on some START->END route."""
        rev = self._reverse_adjacency()
        reach = self._reachable_from(start)
        coreach = self._reachable_to_end(end, rev)
        allowed = reach & coreach
        allowed_edges = {(u, v) for (u, v) in self.edges if u in allowed and v in allowed}
        return allowed, allowed_edges

    @staticmethod
    def _is_valid_path(path, edge_set):
        if not path or len(path) < 2:
            return False
        for i in range(len(path) - 1):
            if (path[i], path[i + 1]) not in edge_set:
                return False
        return True

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
                if path.count(neighbor) < 2:
                    self._dfs(neighbor, end, path)

        path.pop()

    def _extract_basis_set_from_enumerated_paths(self):
        # Cyclomatic complexity computed on the full parsed graph (unchanged behavior)
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
    def _build_forward_tree_parents(self, start, allowed):
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
        next_hop = {end: None}
        q = deque([end])
        while q:
            u = q.popleft()
            for p in rev[u]:
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
        if self.verbose:
            print("[INFO] Using linear (spanning-tree) algorithm")

        allowed, allowed_edges = self.relevant_subgraph(start, end)
        if start not in allowed or end not in allowed:
            print("Error: No paths found from start to end.")
            sys.exit(1)

        rev = self._reverse_adjacency()
        parent = self._build_forward_tree_parents(start, allowed)
        next_hop = self._build_reverse_tree_next_hop(end, allowed, rev)

        base = self._path_start_to(parent, end)
        if base is None:
            print("Error: No paths found from start to end.")
            sys.exit(1)

        tree_edges = set()
        for node, p in parent.items():
            if p is not None:
                tree_edges.add((p, node))

        Vp = len(allowed)
        Ep = len(allowed_edges)
        expected = Ep - Vp + 2

        if self.verbose:
            print(f"[INFO] Relevant subgraph: nodes={Vp}, edges={Ep}, expected_basis={expected}")

        basis = [base]
        seen_paths = {tuple(base)}

        for (u, v) in sorted(allowed_edges):
            if (u, v) in tree_edges:
                continue

            p1 = self._path_start_to(parent, u)
            p2 = self._path_to_end(next_hop, v)
            if p1 is None or p2 is None:
                continue

            candidate = p1 + [v] + p2[1:]
            t = tuple(candidate)
            if t not in seen_paths:
                seen_paths.add(t)
                basis.append(candidate)

            if len(basis) >= expected:
                break

        # Report using relevant-subgraph counts for linear mode
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

        while lines and lines[-1].strip() == "":
            lines.pop()

        if len(lines[1:-1]) == 0:
            print("Error: No edges defined in the input file.")
            sys.exit(1)

        for line in lines[1:-1]:
            parts = line.split()
            if len(parts) < 2:
                print(f"Error: Edge definition '{line}' is incomplete. Expected format: 'NODE1 NODE2'")
                sys.exit(1)
            if len(parts) > 2:
                print(f"Error: Edge definition '{line}' has too many nodes. Expected format: 'NODE1 NODE2'")
                sys.exit(1)

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
    parser = argparse.ArgumentParser(
        description="Compute a basis set of paths from a control flow graph.",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("filename", help="Input CFG text file")
    parser.add_argument("-t", "--time", action="store_true", help="Print elapsed execution time")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose tracing while running")
    parser.add_argument("--linear", action="store_true",
                        help="Use O(E+V) spanning-tree algorithm instead of path enumeration")
    parser.add_argument("--validate", action="store_true",
                        help="Run both algorithms and validate key invariants (may be slow without --max-*)")
    parser.add_argument("--confirm-basis", action="store_true",
                        help="Strongly confirm the produced path set is a true basis (GF(2) rank check)")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON (schema_version 1.1) to stdout")

    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Stop DFS after this many seconds (enumeration mode only)")
    parser.add_argument("--max-calls", type=int, default=None,
                        help="Stop DFS after this many recursive calls (enumeration mode only)")
    parser.add_argument("--max-paths", type=int, default=None,
                        help="Stop DFS after finding this many paths (enumeration mode only)")
    return parser.parse_args()



def gf2_rank_bitmasks(rows):
    """Compute rank over GF(2) for a list of int bitmasks (Gaussian elimination)."""
    pivots = {}  # pivot_bit_index -> row mask
    rank = 0
    for r in rows:
        x = r
        while x:
            p = x.bit_length() - 1  # index of highest-set bit
            if p in pivots:
                x ^= pivots[p]
            else:
                pivots[p] = x
                rank += 1
                break
    return rank


def build_edge_index(allowed_edges):
    """Return (edge_list, edge_to_index) with a stable order."""
    edge_list = sorted(allowed_edges)
    edge_to_index = {e: i for i, e in enumerate(edge_list)}
    return edge_list, edge_to_index


def path_to_incidence_mask(path, edge_to_index):
    """Convert a path to an edge-incidence bitmask (mod 2)."""
    mask = 0
    for i in range(len(path) - 1):
        e = (path[i], path[i + 1])
        idx = edge_to_index.get(e)
        if idx is None:
            return None  # invalid edge for this relevant-subgraph index
        mask ^= (1 << idx)  # XOR implements mod-2 parity
    return mask


def confirm_basis(cfg, start, end, basis_paths, label):
    """Strongly confirm that basis_paths form a true basis using GF(2) rank."""
    allowed_nodes, allowed_edges = cfg.relevant_subgraph(start, end)
    Vp = len(allowed_nodes)
    Ep = len(allowed_edges)
    expected = Ep - Vp + 2

    edge_list, edge_to_index = build_edge_index(allowed_edges)

    # Basic checks: valid paths and build incidence rows
    rows = []
    ok = True
    for k, p in enumerate(basis_paths, 1):
        if not p or p[0] != start or p[-1] != end:
            print(f"[FAIL] {label}: basis path {k} does not start at START and end at END.")
            ok = False
            continue
        m = path_to_incidence_mask(p, edge_to_index)
        if m is None:
            print(f"[FAIL] {label}: basis path {k} contains an edge not in the relevant START→END subgraph.")
            ok = False
            continue
        rows.append(m)

    rank = gf2_rank_bitmasks(rows)
    size_ok = (len(basis_paths) == expected)
    rank_ok = (rank == expected)

    print(f"=== CONFIRM-BASIS ({label}) ===")
    print(f"Relevant subgraph nodes: {Vp}")
    print(f"Relevant subgraph edges: {Ep}")
    print(f"Cyclomatic Complexity (Expected Basis Paths): {expected}")
    print(f"Basis Paths Provided: {len(basis_paths)}")
    print(f"GF(2) Rank of Incidence Vectors: {rank}")

    if not size_ok:
        print(f"Result: FAIL (basis size {len(basis_paths)} != expected {expected})")
        return False, expected, rank
    if not ok:
        print("Result: FAIL (one or more basis paths are invalid)")
        return False, expected, rank
    if not rank_ok:
        print("Result: FAIL (paths are linearly dependent; not a true basis)")
        return False, expected, rank

    print("Result: PASS (valid basis set; independent and correct size)")
    return True, expected, rank


def _print_summary(label, nodes, edges, cyclo, basis_len):
    print(f"=== {label} ===")
    print(f"Nodes: {nodes}")
    print(f"Edges: {edges}")
    print(f"Cyclomatic Complexity (Expected Basis Paths): {cyclo}")
    print(f"Basis Paths Found: {basis_len}")


if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()

    try:
        cfg, start, end = ControlFlowGraph.from_file(args.filename)

        # Apply enumeration search limits (used by the DFS enumeration algorithm)
        cfg.max_seconds = args.max_seconds
        cfg.max_calls = args.max_calls
        cfg.max_paths = args.max_paths

        # JSON mode: suppress human-readable printing and emit one JSON object.
        if args.json:
            if args.validate:
                # Compute expected complexity on the relevant subgraph (same for both algorithms)
                allowed_nodes, allowed_edges = cfg.relevant_subgraph(start, end)
                Vp = len(allowed_nodes)
                Ep = len(allowed_edges)
                expected = Ep - Vp + 2

                # Linear analysis
                basis_linear = cfg.find_basis_set_linear(start, end)
                linear_analysis = _analysis_dict(cfg, start, end, basis_linear, "linear", args.confirm_basis)

                # Enumeration analysis
                enum_completed = False
                enum_error = None
                basis_enum = None
                enum_status = "OK"
                stop_reason = "none"

                try:
                    basis_enum = cfg.find_basis_set(start, end)
                    enum_completed = True
                except SearchLimitReached as e:
                    enum_completed = False
                    enum_error = str(e)
                    enum_status = "INCOMPLETE"
                    # crude stop reason inference
                    msg = str(e).lower()
                    if "seconds" in msg or "time" in msg:
                        stop_reason = "time limit"
                    elif "calls" in msg:
                        stop_reason = "call limit"
                    elif "paths" in msg:
                        stop_reason = "path limit"
                    else:
                        stop_reason = "limit"

                if enum_completed:
                    enum_analysis = _analysis_dict(cfg, start, end, basis_enum, "enumeration", args.confirm_basis)
                    enum_analysis["status"] = "OK"
                    enum_analysis["stop_reason"] = "none"
                else:
                    enum_analysis = _analysis_dict(cfg, start, end, [], "enumeration", args.confirm_basis)
                    enum_analysis["status"] = "INCOMPLETE"
                    enum_analysis["stop_reason"] = stop_reason
                    enum_analysis["error"] = enum_error

                # Validation agreement
                ok_linear = (len(basis_linear) == expected)
                ok_enum = enum_completed and (basis_enum is not None) and (len(basis_enum) == expected)

                agree_cyclo = True  # both use same relevant-subgraph formula
                agree_basis = enum_completed and (len(basis_enum) == len(basis_linear))

                agree_confirm = None
                if args.confirm_basis and enum_completed:
                    agree_confirm = (linear_analysis["confirm_basis"]["pass"] == enum_analysis["confirm_basis"]["pass"])
                elif args.confirm_basis and not enum_completed:
                    agree_confirm = None

                if not ok_linear:
                    validation_status = "FAIL"
                    reason = "linear basis size does not match cyclomatic complexity"
                    exit_code = 1
                elif not enum_completed:
                    validation_status = "INCONCLUSIVE"
                    reason = f"enumeration incomplete ({enum_error})"
                    exit_code = 2
                else:
                    # both completed: decide pass/fail
                    if ok_enum and linear_analysis["confirm_basis"]["pass"] in (True, None) and enum_analysis["confirm_basis"]["pass"] in (True, None):
                        validation_status = "PASS"
                        reason = "both algorithms produced valid bases"
                        exit_code = 0
                    else:
                        validation_status = "FAIL"
                        reason = "enumeration basis failed invariants or confirm-basis"
                        exit_code = 1

                analysis_obj = {
                    "linear": linear_analysis,
                    "enumeration": enum_analysis,
                    "validation": {
                        "status": validation_status,
                        "reason": reason,
                        "agreement": {
                            "cyclomatic_complexity": agree_cyclo,
                            "basis_count": agree_basis,
                            "confirm_basis_pass": agree_confirm
                        }
                    }
                }

                elapsed = time.time() - start_time
                payload = make_json_payload(
                    args=args, cfg=cfg, start=start, end=end,
                    mode="validate", analysis_obj=analysis_obj,
                    exit_code=exit_code,
                    status=("OK" if validation_status == "PASS" else ("INCONCLUSIVE" if validation_status == "INCONCLUSIVE" else "FAIL")),
                    message=f"Validation result: {validation_status}",
                    elapsed_seconds=elapsed
                )
                print(json.dumps(payload, indent=2))
                sys.exit(exit_code)

            else:
                # Single run
                if args.linear:
                    basis_set = cfg.find_basis_set_linear(start, end)
                    algorithm = "linear"
                else:
                    try:
                        basis_set = cfg.find_basis_set(start, end)
                        algorithm = "enumeration"
                    except SearchLimitReached as e:
                        elapsed = time.time() - start_time
                        # Emit JSON with empty basis and ERROR status
                        analysis_obj = _analysis_dict(cfg, start, end, [], "enumeration", args.confirm_basis)
                        payload = make_json_payload(
                            args=args, cfg=cfg, start=start, end=end,
                            mode="single", analysis_obj=analysis_obj,
                            exit_code=2, status="INCONCLUSIVE",
                            message=f"Stopped early due to search limits: {e}",
                            elapsed_seconds=elapsed
                        )
                        print(json.dumps(payload, indent=2))
                        sys.exit(2)

                analysis_obj = _analysis_dict(cfg, start, end, basis_set, algorithm, args.confirm_basis)

                elapsed = time.time() - start_time
                exit_code = 0
                status = "OK"
                message = "Basis paths computed successfully"
                payload = make_json_payload(
                    args=args, cfg=cfg, start=start, end=end,
                    mode="single", analysis_obj=analysis_obj,
                    exit_code=exit_code, status=status, message=message,
                    elapsed_seconds=elapsed
                )
                print(json.dumps(payload, indent=2))
                sys.exit(0)

        # Non-JSON mode (original behavior)
        if args.validate:
            # (existing validate logic preserved)
            # Compute expected complexity on the relevant subgraph
            allowed, allowed_edges = cfg.relevant_subgraph(start, end)
            Vp = len(allowed)
            Ep = len(allowed_edges)
            expected = Ep - Vp + 2

            # 1) Linear algorithm first (fast)
            basis_linear = cfg.find_basis_set_linear(start, end)

            # Validate linear invariants
            ok_linear = True
            if len(basis_linear) != expected:
                print(f"[FAIL] Linear basis size {len(basis_linear)} != expected {expected}")
                ok_linear = False

            edge_set = cfg.edges
            for i, p in enumerate(basis_linear, 1):
                if not (p and p[0] == start and p[-1] == end and ControlFlowGraph._is_valid_path(p, edge_set)):
                    print(f"[FAIL] Linear basis path {i} is not a valid START->END path.")
                    ok_linear = False
                    break

            # Always show linear summary
            _print_summary("LINEAR ALGORITHM (relevant subgraph)", Vp, Ep, expected, len(basis_linear))

            if args.confirm_basis:
                ok_basis_linear, _, _ = confirm_basis(cfg, start, end, basis_linear, "LINEAR")
                if not ok_basis_linear:
                    print("Validation result: FAIL (linear basis failed --confirm-basis)")
                    sys.exit(1)

            # 2) Enumeration algorithm (may be slow). Catch limits and report INCONCLUSIVE.
            enum_completed = False
            enum_error = None
            basis_enum = None

            try:
                basis_enum = cfg.find_basis_set(start, end)
                enum_completed = True
            except SearchLimitReached as e:
                enum_completed = False
                enum_error = str(e)

            if not enum_completed:
                print("=== ENUMERATION ALGORITHM ===")
                print(f"Status: INCOMPLETE ({enum_error})")
                print(f"Progress so far: calls={cfg.dfs_calls}, paths_found={len(cfg.paths)}")
                if ok_linear:
                    print("Validation result: INCONCLUSIVE (enumeration did not complete within limits)")
                else:
                    print("Validation result: FAIL (linear invariants failed; enumeration did not complete)")
                sys.exit(1 if not ok_linear else 2)

            # Enumeration completed: validate and show summary
            ok = ok_linear
            if len(basis_enum) != expected:
                print(f"[FAIL] Enumeration basis size {len(basis_enum)} != expected {expected}")
                ok = False

            for i, p in enumerate(basis_enum, 1):
                if not (p and p[0] == start and p[-1] == end and ControlFlowGraph._is_valid_path(p, edge_set)):
                    print(f"[FAIL] Enumeration basis path {i} is not a valid START->END path.")
                    ok = False
                    break

            _print_summary("ENUMERATION ALGORITHM (relevant expected)", Vp, Ep, expected, len(basis_enum))

            if args.confirm_basis:
                ok_basis_enum, _, _ = confirm_basis(cfg, start, end, basis_enum, "ENUMERATION")
                if not ok_basis_enum:
                    ok = False

            if ok:
                print("Validation result: PASS")
                sys.exit(0)
            else:
                print("Validation result: FAIL")
                sys.exit(1)

        # Normal single-algorithm run
        if args.linear:
            basis_set = cfg.find_basis_set_linear(start, end)
        else:
            try:
                basis_set = cfg.find_basis_set(start, end)
            except SearchLimitReached as e:
                print(f"Stopped early due to search limits: {e}")
                print(f"Progress so far: calls={cfg.dfs_calls}, paths_found={len(cfg.paths)}")
                sys.exit(2)

        print("Basis Set of Paths:")
        for path in basis_set:
            print(" -> ".join(path))

        # Print the number of basis paths, nodes, edges, and computed cyclomatic complexity
        print(f"Number of Nodes: {cfg.num_nodes}")
        print(f"Number of Edges: {cfg.num_edges}")
        print(f"Cyclomatic Complexity (Expected Basis Paths): {cfg.cyclomatic_complexity}")
        print(f"Number of Basis Paths Found: {cfg.basis_path_count}")

        if args.confirm_basis:
            ok_basis, _, _ = confirm_basis(cfg, start, end, basis_set, "RUN")
            if not ok_basis:
                sys.exit(1)

        if args.time:
            elapsed = time.time() - start_time
            print(f"Elapsed Time (seconds): {elapsed:.6f}")

        sys.exit(0)

    except KeyboardInterrupt:
        # Preserve existing behavior for Ctrl+C
        print("Error: Interrupted by user.")
        sys.exit(130)
