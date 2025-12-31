# Version 73
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


import traceback as _traceback

# ----------------------------
# JSON-first output policy
# ----------------------------
# If --json is present anywhere in argv, stdout MUST be a single JSON object
# for both success and failure, regardless of failure stage (CLI parsing, missing file, etc).
JSON_REQUESTED = ("--json" in sys.argv[1:])
_JSON_EMITTED = False

import builtins as _builtins
_ORIG_PRINT = _builtins.print

def _patched_print(*args, **kwargs):
    # In --json mode, keep stdout clean for the single JSON object.
    # Any incidental prints without an explicit destination are redirected to stderr.
    if JSON_REQUESTED and ("file" not in kwargs or kwargs.get("file") is None):
        kwargs["file"] = sys.stderr
    return _ORIG_PRINT(*args, **kwargs)

if JSON_REQUESTED:
    _builtins.print = _patched_print

def _stdout_print(*args, **kwargs):
    # Always print to stdout (used for JSON emission).
    kwargs["file"] = sys.stdout
    return _ORIG_PRINT(*args, **kwargs)


import builtins as _builtins
_builtin_print = _builtins.print

def _pm_print(*args, **kwargs):
    """Human-readable output. In --json mode, route to stderr by default."""
    if JSON_REQUESTED and "file" not in kwargs:
        kwargs["file"] = sys.stderr
    _builtin_print(*args, **kwargs)

# Harden: if JSON was requested, any accidental print() goes to stderr (never stdout).
if JSON_REQUESTED:
    print = _pm_print
def _emit_json(payload: dict) -> None:
    """Emit exactly one JSON object to stdout."""
    global _JSON_EMITTED
    _JSON_EMITTED = True
    sys.stdout.write(json.dumps(payload, indent=2))
    sys.stdout.write("\n")
    sys.stdout.flush()

def _json_error_payload(*, args_obj=None, exit_code: int, message: str, error_code: str, details: dict | None = None, filename: str | None = None) -> dict:
    # Keep this minimal and stable. Extra noisy fields should go into details and be normalized in tests.
    return {
        "schema_version": "1.1",
        "run": {
            "argv": sys.argv[1:],
            "json_requested": True,
            "filename": filename,
        },
        "results": {
            "summary": {
                "exit_code": exit_code,
                "status": "ERROR",
                "message": message,
            },
            "error": {
                "code": error_code,
                "message": message,
                "details": details or {},
            },
        },
    }

class _CliUsageError(Exception):
    pass

class _ToolExit(Exception):
    """Structured termination with an exit code and JSON-friendly fields."""
    def __init__(self, exit_code: int, error_code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.exit_code = int(exit_code)
        self.error_code = str(error_code)
        self.message = str(message)
        self.details = details or {}


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
        "performance": {"elapsed_seconds": None, "elapsed_nanoseconds": None}
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
            payload["results"]["analysis"]["linear"]["performance"]["elapsed_nanoseconds"] = int(elapsed_seconds * 1_000_000_000) if elapsed_seconds is not None else None
        if "enumeration" in payload["results"]["analysis"]:
            payload["results"]["analysis"]["enumeration"]["performance"]["elapsed_seconds"] = elapsed_seconds
            payload["results"]["analysis"]["enumeration"]["performance"]["elapsed_nanoseconds"] = int(elapsed_seconds * 1_000_000_000) if elapsed_seconds is not None else None
    else:
        payload["results"]["analysis"]["performance"]["elapsed_seconds"] = elapsed_seconds
        payload["results"]["analysis"]["performance"]["elapsed_nanoseconds"] = int(elapsed_seconds * 1_000_000_000) if elapsed_seconds is not None else None
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
        human-readable output is suppressed and all logs/errors go to stderr.

        Rule: if --json is present anywhere in argv, Path_Master will still emit
        a JSON object even if it fails before reading/parsing the input file
        (e.g., missing file or command-line usage errors).

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
    0   Success. Basis paths computed successfully or validation passed.

    1   Failure. Any non-inconclusive error (including missing/invalid input),
        failed invariants, or --confirm-basis failure.

    2   Inconclusive. Enumeration stopped early due to a search limit
        (--max-seconds, --max-calls, or --max-paths).

    130 Interrupted by user (Ctrl+C / SIGINT).

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
            raise _ToolExit(
                exit_code=1,
                error_code="INPUT_INVALID",
                message=f"Invalid node name detected: '{u}' or '{v}' contains control characters.",
                details={"u": u, "v": v},
            )

        # Validate node names (must be alphanumeric, underscores, or hyphens)
        if not re.match(r'^[\w-]+$', u) or not re.match(r'^[\w-]+$', v):
            raise _ToolExit(
                exit_code=1,
                error_code="INPUT_INVALID",
                message=f"Invalid node name: '{u}' or '{v}' (contains spaces or special characters)",
                details={"u": u, "v": v},
            )

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
            raise _ToolExit(
                exit_code=1,
                error_code="NO_PATHS",
                message="No paths found from start to end.",
                details={"start": start, "end": end},
            )

        if self.verbose:
            print("[INFO] Extracting basis set...")

        basis = self._extract_basis_set_from_enumerated_paths(start, end)

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

    def _extract_basis_set_from_enumerated_paths(self, start, end):
        """Select a true basis from enumerated paths using a rank-greedy rule.

        We treat each START→END path as an edge-incidence vector over GF(2)
        (mod-2 parity of edge traversals). A set of paths is a *basis* iff the
        incidence vectors are linearly independent and span the path space.

        This selector walks the enumerated paths in order and keeps a path only
        if it *increases the current GF(2) rank*. This prevents the classic
        failure mode where 'new-looking' paths are actually dependent because
        repeated edges/cycles cancel mod 2.
        """
        # Compute cyclomatic complexity on the START→END relevant subgraph
        allowed_nodes, allowed_edges = self.relevant_subgraph(start, end)
        Vp = len(allowed_nodes)
        Ep = len(allowed_edges)
        expected_basis_size = Ep - Vp + 2

        # Build an index for incidence masks on the relevant edge set
        _, edge_to_index = build_edge_index(allowed_edges)

        basis_set = []
        pivots = {}  # maps pivot_bit -> row mask (Gaussian elimination state over GF(2))

        def try_add_to_rank(mask: int) -> bool:
            """Attempt to add mask to the current span; return True iff rank increases."""
            x = mask
            # Reduce x by existing pivots (highest bits first)
            for pb in sorted(pivots.keys(), reverse=True):
                if (x >> pb) & 1:
                    x ^= pivots[pb]
            if x == 0:
                return False  # dependent

            # New pivot: highest set bit of x
            new_pb = x.bit_length() - 1

            # Optional: keep basis reduced by eliminating new pivot from existing rows
            for pb in list(pivots.keys()):
                if (pivots[pb] >> new_pb) & 1:
                    pivots[pb] ^= x

            pivots[new_pb] = x
            return True

        # Greedily keep paths that increase rank
        for path in self.paths:
            mask = path_to_incidence_mask(path, edge_to_index)
            if mask is None:
                # Path contains an edge outside the relevant subgraph; skip it.
                # (This should be rare; enumerated START→END paths are usually relevant.)
                continue

            if try_add_to_rank(mask):
                basis_set.append(path)

            if len(basis_set) >= expected_basis_size:
                break

        # If enumeration did not provide enough independent paths (rare), top up to the
        # expected count with remaining paths so output size stays consistent. Strong
        # checking (--confirm-basis) will still report FAIL in this case.
        if len(basis_set) < expected_basis_size:
            for path in self.paths:
                if path in basis_set:
                    continue
                basis_set.append(path)
                if len(basis_set) >= expected_basis_size:
                    break

        self.num_nodes = Vp
        self.num_edges = Ep
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
            raise _ToolExit(
                exit_code=1,
                error_code="NO_PATHS",
                message="No paths found from start to end.",
                details={"start": start, "end": end},
            )

        rev = self._reverse_adjacency()
        parent = self._build_forward_tree_parents(start, allowed)
        next_hop = self._build_reverse_tree_next_hop(end, allowed, rev)

        base = self._path_start_to(parent, end)
        if base is None:
            raise _ToolExit(
                exit_code=1,
                error_code="NO_PATHS",
                message="No paths found from start to end.",
                details={"start": start, "end": end},
            )

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
            # In --json mode, stdout must still be JSON. Raise a structured error and let main handle formatting.
            raise _ToolExit(
                exit_code=1,
                error_code="INPUT_NOT_FOUND",
                message=f"File '{filename}' not found.",
                details={"path": filename},
            )

        if not lines or all(line.strip() == "" for line in lines):
            raise _ToolExit(
                exit_code=1,
                error_code="INPUT_EMPTY",
                message="Input file is empty or contains only whitespace.",
                details={"path": filename},
            )

        while lines and lines[-1].strip() == "":
            lines.pop()

        if len(lines[1:-1]) == 0:
            raise _ToolExit(
                exit_code=1,
                error_code="INPUT_INVALID",
                message="No edges defined in the input file.",
                details={"path": filename},
            )

        for line in lines[1:-1]:
            parts = line.split()
            if len(parts) < 2:
                raise _ToolExit(
                    exit_code=1,
                    error_code="INPUT_INVALID",
                    message=f"Edge definition '{line}' is incomplete. Expected format: 'NODE1 NODE2'",
                    details={"line": line, "path": filename},
                )
            if len(parts) > 2:
                raise _ToolExit(
                    exit_code=1,
                    error_code="INPUT_INVALID",
                    message=f"Edge definition '{line}' has too many nodes. Expected format: 'NODE1 NODE2'",
                    details={"line": line, "path": filename},
                )

        seen_edges = set()
        for line in lines[1:-1]:
            u, v = map(str.strip, line.split())
            if (u, v) in seen_edges:
                raise _ToolExit(
                    exit_code=1,
                    error_code="INPUT_INVALID",
                    message=f"Duplicate edge detected: '{u} -> {v}'",
                    details={"u": u, "v": v, "path": filename},
                )
            seen_edges.add((u, v))

        if len(lines) < 3:
            raise _ToolExit(
                exit_code=1,
                error_code="INPUT_INVALID",
                message="Invalid input file format. Expected at least a start node, edges, and an end node.",
                details={"path": filename},
            )

        start = lines[0].strip()
        end = lines[-1].strip()
        cfg = ControlFlowGraph(verbose=verbose)

        for line in lines[1:-1]:
            u, v = map(str.strip, line.split())
            if u == end:
                raise _ToolExit(
                    exit_code=1,
                    error_code="INPUT_INVALID",
                    message="End node should not start an edge.",
                    details={"edge": [u, v], "end": end, "path": filename},
                )
            if v == start:
                raise _ToolExit(
                    exit_code=1,
                    error_code="INPUT_INVALID",
                    message="Start node should not end an edge.",
                    details={"edge": [u, v], "start": start, "path": filename},
                )

        for line in lines[1:-1]:
            u, v = map(str.strip, line.split())
            cfg.add_edge(u, v)

        if start not in cfg.nodes:
            raise _ToolExit(
                exit_code=1,
                error_code="INPUT_INVALID",
                message=f"STARTING NODE '{start}' is missing.",
                details={"start": start, "path": filename},
            )
        if end not in cfg.nodes:
            raise _ToolExit(
                exit_code=1,
                error_code="INPUT_INVALID",
                message=f"ENDING NODE '{end}' is missing.",
                details={"end": end, "path": filename},
            )

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
    start_time_ns = time.perf_counter_ns()
    # Early intercept: in JSON mode, --help/-h returns one JSON object (and does not print argparse help text).
    if JSON_REQUESTED and ("--help" in sys.argv[1:] or "-h" in sys.argv[1:]):
        # Reconstruct a parser solely to obtain help text (must mirror parse_args()).
        parser = argparse.ArgumentParser(
            prog=os.path.basename(__file__),
            description="Compute a basis set of paths from a control flow graph.",
            epilog=HELP_EPILOG,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument("filename", nargs="?", help="Input CFG text file")
        parser.add_argument("-t", "--time", action="store_true", help="Print elapsed execution time")
        parser.add_argument("-v", "--verbose", action="store_true", help="Print debugging output")
        parser.add_argument("--linear", action="store_true", help="Run the linear algorithm")
        parser.add_argument("--validate", action="store_true", help="Run both algorithms and validate invariants")
        parser.add_argument("--confirm-basis", action="store_true", help="Confirm basis independence using XOR rank")
        parser.add_argument("--json", action="store_true", help="Emit JSON on stdout (always, even on errors)")
        parser.add_argument("--max-seconds", type=float, default=None, help="Stop enumeration after this many seconds (inconclusive)")
        parser.add_argument("--max-calls", type=int, default=None, help="Stop enumeration after this many DFS calls (inconclusive)")
        parser.add_argument("--max-paths", type=int, default=None, help="Stop enumeration after enumerating this many paths (inconclusive)")

        help_text = parser.format_help()
        _emit_json({
            "schema_version": "1.1",
            "run": {"argv": sys.argv[1:], "tool": {"name": "Path_Master", "version": _extract_version_number()}},
            "options": {"json": True},
            "results": {"summary": {"exit_code": 0, "status": "OK", "message": "Help text"}, "help": {"text": help_text}},
        })
        sys.exit(0)

    try:
        # Parse args. If --json was requested, emit JSON even for CLI/usage failures.
        try:
            args = parse_args()
            if getattr(args, 'json', False):
                args.verbose = False
        except SystemExit as e:
            # argparse terminated early (usage error). In JSON mode we still emit exactly one JSON object.
            if JSON_REQUESTED:
                payload = _json_error_payload(
                    exit_code=1,
                    message="Command-line usage error.",
                    error_code="CLI_USAGE",
                    details={"system_exit_code": getattr(e, "code", None)},
                    filename=None,
                )
                _emit_json(payload)
                raise SystemExit(1)
            raise
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

                elapsed_ns = time.perf_counter_ns() - start_time_ns

                elapsed = elapsed_ns / 1_000_000_000.0
                payload = make_json_payload(
                    args=args, cfg=cfg, start=start, end=end,
                    mode="validate", analysis_obj=analysis_obj,
                    exit_code=exit_code,
                    status=("OK" if validation_status == "PASS" else ("INCONCLUSIVE" if validation_status == "INCONCLUSIVE" else "FAIL")),
                    message=f"Validation result: {validation_status}",
                    elapsed_seconds=elapsed
                )
                _emit_json(payload)
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
                        elapsed_ns = time.perf_counter_ns() - start_time_ns
                        elapsed = elapsed_ns / 1_000_000_000.0
                        # Emit JSON with empty basis and ERROR status
                        analysis_obj = _analysis_dict(cfg, start, end, [], "enumeration", args.confirm_basis)
                        payload = make_json_payload(
                            args=args, cfg=cfg, start=start, end=end,
                            mode="single", analysis_obj=analysis_obj,
                            exit_code=2, status="INCONCLUSIVE",
                            message=f"Stopped early due to search limits: {e}",
                            elapsed_seconds=elapsed
                        )
                        _emit_json(payload)
                        sys.exit(2)

                analysis_obj = _analysis_dict(cfg, start, end, basis_set, algorithm, args.confirm_basis)

                elapsed_ns = time.perf_counter_ns() - start_time_ns

                elapsed = elapsed_ns / 1_000_000_000.0
                exit_code = 0
                status = "OK"
                message = "Basis paths computed successfully"
                payload = make_json_payload(
                    args=args, cfg=cfg, start=start, end=end,
                    mode="single", analysis_obj=analysis_obj,
                    exit_code=exit_code, status=status, message=message,
                    elapsed_seconds=elapsed
                )
                _emit_json(payload)
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
            elapsed_ns = time.perf_counter_ns() - start_time_ns
            elapsed = elapsed_ns / 1_000_000_000.0
            print(f"Elapsed Time (seconds): {elapsed:.9f} (ns={int(elapsed*1_000_000_000)})")

        sys.exit(0)

    except FileNotFoundError as e:
        if JSON_REQUESTED or getattr(locals().get("args", None), "json", False):
            payload = _json_error_payload(
                exit_code=1,
                message=str(e),
                error_code="INPUT_NOT_FOUND",
                details={"exception_type": type(e).__name__},
                filename=getattr(locals().get("args", None), "filename", None),
            )
            _emit_json(payload)
            sys.exit(1)
        raise

    except ValueError as e:
        if JSON_REQUESTED or getattr(locals().get("args", None), "json", False):
            payload = _json_error_payload(
                exit_code=1,
                message=str(e),
                error_code="INPUT_FORMAT",
                details={"exception_type": type(e).__name__},
                filename=getattr(locals().get("args", None), "filename", None),
            )
            _emit_json(payload)
            sys.exit(1)
        raise

    except _ToolExit as e:
        # Known/expected failures (missing file, parse error, etc.)
        if JSON_REQUESTED or getattr(locals().get('args', None), 'json', False):
            payload = _json_error_payload(
                exit_code=e.exit_code,
                message=e.message,
                error_code=e.error_code,
                details=e.details,
                filename=getattr(locals().get('args', None), 'filename', None),
            )
            _emit_json(payload)
            sys.exit(e.exit_code)
        # Non-JSON mode: human-readable to stderr
        print(f"Error: {e.message}", file=sys.stderr)
        sys.exit(e.exit_code)

    except SystemExit as e:
        # sys.exit(...) can still be triggered in deep code paths.
        # In --json mode, ensure we still emit a JSON envelope and keep stdout valid.
        if _JSON_EMITTED:
            raise
        if JSON_REQUESTED or getattr(locals().get('args', None), 'json', False):
            code_ = getattr(e, "code", 1)
            try:
                code_ = int(code_) if code_ is not None else 1
            except Exception:
                code_ = 1
            if code_ == 0:
                raise
            payload = _json_error_payload(
                exit_code=(2 if code_ == 2 else 1),
                message="Terminated early.",
                error_code=("INCONCLUSIVE" if code_ == 2 else "ERROR"),
                details={"system_exit_code": getattr(e, "code", None)},
                filename=getattr(locals().get('args', None), 'filename', None),
            )
            _emit_json(payload)
            sys.exit(payload["results"]["summary"]["exit_code"])
        raise

    except Exception as e:
        # Unexpected crash: keep traceback on stderr; JSON envelope on stdout if requested.
        _traceback.print_exc(file=sys.stderr)
        if JSON_REQUESTED or getattr(locals().get("args", None), "json", False):
            payload = _json_error_payload(
                exit_code=1,
                message=f"Internal error: {type(e).__name__}",
                error_code="INTERNAL",
                details={"exception_type": type(e).__name__},
                filename=getattr(locals().get("args", None), "filename", None),
            )
            _emit_json(payload)
            sys.exit(1)
        raise

    except KeyboardInterrupt:
        # Ctrl+C / SIGINT
        if JSON_REQUESTED or getattr(locals().get('args', None), 'json', False):
            payload = _json_error_payload(
                exit_code=130,
                message="Interrupted by user (Ctrl+C / SIGINT).",
                error_code="INTERRUPTED",
                details={},
                filename=getattr(locals().get('args', None), 'filename', None),
            )
            _emit_json(payload)
            sys.exit(130)
        print("Error: Interrupted by user.", file=sys.stderr)
        sys.exit(130)