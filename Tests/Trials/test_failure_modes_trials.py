import json
import subprocess
from pathlib import Path
from typing import Any, Optional

import pytest

# Project root: Tests/trials/ -> project root
ROOT = Path(__file__).resolve().parents[2]
TESTDATA = ROOT / "TestData"

# Cases driven from Some_Failure_Mode_Trials.txt (updated):
# - Most cases are expected to FAIL (non-zero rc, error-ish status, no basis paths).
# - Trailing_Whitespace.txt is expected to SUCCEED (rc=0, success status, basis paths generated).
FAILURE_CASES = [
    ("FM-001", "What if the END node is not connected to the directed graph using PathMaster.py in both enumerated or linear algorithm? Confirm error message is generated, confirm error return code is generated. Confirm basis paths are not generated.", "Disconnected_from_END.txt", "FAIL"),
    ("FM-002", "What if the START node is not connected to the directed graph using PathMaster.py in both enumerated or linear algorithm? Confirm error message is generated, confirm error return code is generated. Confirm basis paths are not generated.", "Disconnected_from_START.txt", "FAIL"),
    ("FM-003", "What if the same edge is duplicated in the directed graph using PathMaster.py in both enumerated or linear algorithm? Confirm error message is generated, confirm error return code is generated. Confirm basis paths are not generated.", "Duplicate_Edges.txt", "FAIL"),
    ("FM-004", "What if the End node is in a loop in the directed graph using PathMaster.py in both enumerated or linear algorithm? Confirm error message is generated, confirm error return code is generated. Confirm basis paths are not generated.", "End_In_Loop.txt", "FAIL"),
    ("FM-005", "What if the input file contains only white space using PathMaster.py in both enumerated or linear algorithm? Confirm error message is generated, confirm error return code is generated. Confirm basis paths are not generated.", "File_Contains_Only_Whitespace.txt", "FAIL"),
    ("FM-006", "What if the input file is empty using PathMaster.py in both enumerated or linear algorithm? Confirm error message is generated, confirm error return code is generated. Confirm basis paths are not generated.", "File_Is_Empty.txt", "FAIL"),
    ("FM-007", "What if the input file includes an invalid edge which has too few nodes in it using PathMaster.py in both enumerated or linear algorithm? Confirm error message is generated, confirm error return code is generated. Confirm basis paths are not generated.", "Invalid_Edge_Formatting_too_few_nodes.txt", "FAIL"),
    ("FM-008", "What if the input file includes an invalid edge which has too many nodes in it using PathMaster.py in both enumerated or linear algorithm? Confirm error message is generated, confirm error return code is generated. Confirm basis paths are not generated.", "Invalid_Edge_Formatting_too_many_nodes.txt", "FAIL"),
    ("FM-009", "What if the input file includes an invalid character in a node using PathMaster.py in both enumerated or linear algorithm? Confirm error message is generated, confirm error return code is generated. Confirm basis paths are not generated.", "invalid_node_char.txt", "FAIL"),
    ("FM-010", "What if the input file does not define any edges using PathMaster.py in both enumerated or linear algorithm? Confirm error message is generated, confirm error return code is generated. Confirm basis paths are not generated.", "missing_edges.txt", "FAIL"),
    ("FM-011", "What if the input file does not define an End node using PathMaster.py in both enumerated or linear algorithm? Confirm error message is generated, confirm error return code is generated. Confirm basis paths are not generated.", "missing_end.txt", "FAIL"),
    ("FM-012", "What if the input file does not define a Start node using PathMaster.py in both enumerated or linear algorithm? Confirm error message is generated, confirm error return code is generated. Confirm basis paths are not generated.", "missing_start.txt", "FAIL"),
    ("FM-013", "What if the input file does not define a Start node and does not define an End node using PathMaster.py in both enumerated or linear algorithm? Confirm error message is generated, confirm error return code is generated. Confirm basis paths are not generated.", "missing_start_and_end.txt", "FAIL"),
    ("FM-014", "What if the input file defines a directed graph with both a Start Node in a loop and an End Node in a loop using PathMaster.py in both enumerated or linear algorithm? Confirm error message is generated, confirm error return code is generated. Confirm basis paths are not generated.", "Start_and_End_In_Loop.txt", "FAIL"),
    ("FM-015", "What if the input file defines a directed graph with a Start Node in a loop using PathMaster.py in both enumerated or linear algorithm? Confirm error message is generated, confirm error return code is generated. Confirm basis paths are not generated.", "Start_In_Loop.txt", "FAIL"),

    # Updated expectation: should NOT fail
    ("FM-016", "What if the input file defines a well defined directed graph followed by several lines of white space using PathMaster.py in both enumerated or linear algorithm? Confirm no error message is generated, confirm no error return code is generated. Confirm basis paths are generated.", "Trailing_Whitespace.txt", "PASS"),

    ("FM-017", "What if the input file does not exist using PathMaster.py in both enumerated or linear algorithm? Confirm error message is generated, confirm error return code is generated. Confirm basis paths are not generated.", "No_Such_File.txt", "FAIL"),
    ("FM-018", "What if the input file defines a directed graph with no path from the Start Node to the End Node using PathMaster.py in both enumerated or linear algorithm? Confirm error message is generated, confirm error return code is generated. Confirm basis paths are not generated.", "No_Paths.txt", "FAIL"),
]


def run_path_master_json(tool_path: Path, cfg_file: Path, extra_args=None, timeout_s: int = 5) -> tuple[int, dict, str, str]:
    extra_args = extra_args or []
    cmd = ["python", str(tool_path), "--json", "--confirm-basis", *extra_args, str(cfg_file)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    stdout = (p.stdout or "").strip()
    if not stdout:
        raise AssertionError(f"Path_Master produced empty stdout.\nSTDERR:\n{p.stderr}")
    obj = json.loads(stdout)
    return p.returncode, obj, stdout, (p.stderr or "")


def _get(pm_json: dict, path: list[str], default=None):
    cur = pm_json
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _find_anywhere(obj: Any, predicate, path_prefix: str = "") -> Optional[tuple[str, Any]]:
    """DFS-search any nested structure and return (path, value) for first match."""
    if predicate(obj):
        return (path_prefix or "$", obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            found = _find_anywhere(v, predicate, f"{path_prefix}.{k}" if path_prefix else k)
            if found is not None:
                return found
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            found = _find_anywhere(v, predicate, f"{path_prefix}[{i}]")
            if found is not None:
                return found
    return None


def _extract_basis_paths(pm_json: dict) -> tuple[Optional[Any], Optional[str]]:
    """
    Try to find basis paths even if schema changes.

    Returns (basis_paths, found_at_path).
    """
    # Canonical (what earlier tests assumed)
    for key_path in [
        ["results", "analysis", "basis", "basis_paths"],
        ["results", "analysis", "basis", "paths"],
        ["results", "analysis", "basis", "path_list"],
        ["results", "analysis", "basis", "basis"],
    ]:
        v = _get(pm_json, key_path)
        if v is not None:
            return v, ".".join(key_path)

    # If basis exists, search within it for a list that looks like "paths"
    basis = _get(pm_json, ["results", "analysis", "basis"])
    if isinstance(basis, dict):
        found = _find_anywhere(
            basis,
            lambda x: isinstance(x, list) and len(x) > 0,
            "results.analysis.basis",
        )
        if found is not None:
            return found[1], found[0]

    return None, None


def assert_failure_shape(checks, label: str, rc: int, pm_json: dict):
    """Expected FAIL: non-zero rc, error-ish status/message, and no basis paths."""
    summary = _get(pm_json, ["results", "summary"], {}) or {}
    status = summary.get("status")
    msg = summary.get("message")

    checks.check(rc != 0, f"{label}: process rc is non-zero", details={"rc": rc})
    checks.check(
        status in ("ERROR", "INCONCLUSIVE"),
        f"{label}: JSON status indicates non-success",
        details={"status": status, "message": msg},
    )
    checks.check(bool(msg), f"{label}: JSON summary.message is present", details={"message": msg})

    basis = _get(pm_json, ["results", "analysis", "basis"])
    basis_count = _get(pm_json, ["results", "analysis", "basis", "basis_count"])
    basis_paths, basis_paths_at = _extract_basis_paths(pm_json)

    checks.check(
        (basis is None) or (basis_count in (None, 0)) or (basis_paths in (None, [], {})),
        f"{label}: no basis paths generated",
        details={
            "basis_present": basis is not None,
            "basis_count": basis_count,
            "basis_paths_found": basis_paths is not None,
            "basis_paths_at": basis_paths_at,
            "basis_paths_type": type(basis_paths).__name__ if basis_paths is not None else None,
        },
    )


def assert_success_shape(checks, label: str, rc: int, pm_json: dict):
    """
    Expected PASS: rc==0, success-ish status, basis_count>0, and confirm-basis passes.

    NOTE: Some schemas may not include the full path list under a stable key. We therefore:
      - REQUIRE basis_count>0
      - REQUIRE confirm_basis.pass == True
      - TRY to locate basis paths; if not found, we log a NOTE instead of failing.
    """
    summary = _get(pm_json, ["results", "summary"], {}) or {}
    status = summary.get("status")
    msg = summary.get("message")

    checks.check(rc == 0, f"{label}: process rc is zero", details={"rc": rc})
    checks.check(
        status in ("OK", "SUCCESS"),
        f"{label}: JSON status indicates success",
        details={"status": status, "message": msg},
    )

    basis_count = _get(pm_json, ["results", "analysis", "basis", "basis_count"])
    confirm_pass = _get(pm_json, ["results", "analysis", "confirm_basis", "pass"])

    checks.check(basis_count is not None, f"{label}: basis_count present", details={"basis_count": basis_count})
    checks.check(int(basis_count) > 0, f"{label}: basis_count > 0", details={"basis_count": basis_count})
    checks.check(confirm_pass is True, f"{label}: --confirm-basis passes", details={"confirm_pass": confirm_pass})

    basis_paths, basis_paths_at = _extract_basis_paths(pm_json)
    if basis_paths is None:
        checks.note(
            f"{label}: basis paths not found under expected JSON keys (but basis_count>0 suggests they exist)",
            details={"basis_paths_at": basis_paths_at},
        )
    else:
        checks.note(
            f"{label}: basis paths found",
            details={"basis_paths_at": basis_paths_at, "basis_paths_type": type(basis_paths).__name__},
        )
        if isinstance(basis_paths, list):
            checks.check(len(basis_paths) > 0, f"{label}: basis paths non-empty list", details={"len": len(basis_paths)})

    cyclo = _get(pm_json, ["results", "analysis", "relevant_subgraph", "cyclomatic_complexity"])
    if cyclo is None:
        cyclo = _get(pm_json, ["results", "analysis", "cyclomatic_complexity"])
    if cyclo is not None and basis_count is not None:
        checks.check(
            int(basis_count) == int(cyclo),
            f"{label}: basis_count == cyclomatic_complexity",
            details={"basis_count": basis_count, "cyclomatic_complexity": cyclo},
        )


def _case_param(case_id: str, objective_text: str, filename: str, expectation: str):
    return pytest.param(
        case_id,
        objective_text,
        filename,
        expectation,
        marks=[
            pytest.mark.failure_modes,
            pytest.mark.objective(id=case_id, text=objective_text, ideas=["Failure Modes"]),
        ],
        id=f"{case_id}:{filename}",
    )


@pytest.mark.parametrize(
    "case_id, objective_text, filename, expectation",
    [_case_param(cid, text, fn, exp) for (cid, text, fn, exp) in FAILURE_CASES],
)
def test_failure_modes_enum_and_linear(tool_path, artifacts_dir, checks, case_id, objective_text, filename, expectation):
    cfg_file = TESTDATA / filename

    if filename.lower() == "no_such_file.txt":
        checks.check(not cfg_file.exists(), f"Precondition: file should not exist: {cfg_file}")
    else:
        checks.check(cfg_file.exists(), f"Precondition: input exists: {cfg_file}")

    case_dir = artifacts_dir / f"{case_id}_{cfg_file.stem}"
    case_dir.mkdir(parents=True, exist_ok=True)
    checks.note("Case directory", details={"case_dir": str(case_dir), "input": str(cfg_file), "expectation": expectation})

    # --- Enumeration (default) ---
    try:
        rc_e, pm_e_json, pm_e_stdout, pm_e_stderr = run_path_master_json(Path(tool_path), cfg_file, extra_args=[], timeout_s=5)
    except subprocess.TimeoutExpired:
        checks.check(False, "Path_Master(enum) completed within timeout", details={"timeout_s": 5})
        return

    (case_dir / "path_master_enum.json").write_text(json.dumps(pm_e_json, indent=2), encoding="utf-8")
    (case_dir / "path_master_enum_stdout.json").write_text(pm_e_stdout + "\n", encoding="utf-8")
    (case_dir / "path_master_enum_stderr.txt").write_text(pm_e_stderr, encoding="utf-8")

    # --- Linear ---
    try:
        rc_l, pm_l_json, pm_l_stdout, pm_l_stderr = run_path_master_json(Path(tool_path), cfg_file, extra_args=["--linear"], timeout_s=5)
    except subprocess.TimeoutExpired:
        checks.check(False, "Path_Master(linear) completed within timeout", details={"timeout_s": 5})
        return

    (case_dir / "path_master_linear.json").write_text(json.dumps(pm_l_json, indent=2), encoding="utf-8")
    (case_dir / "path_master_linear_stdout.json").write_text(pm_l_stdout + "\n", encoding="utf-8")
    (case_dir / "path_master_linear_stderr.txt").write_text(pm_l_stderr, encoding="utf-8")

    if expectation == "FAIL":
        assert_failure_shape(checks, "enum", rc_e, pm_e_json)
        assert_failure_shape(checks, "linear", rc_l, pm_l_json)
        checks.check(
            rc_e != 0 and rc_l != 0,
            "both enum and linear runs returned non-zero rc (expected for FAIL cases)",
            details={"rc_enum": rc_e, "rc_linear": rc_l},
        )
    else:
        assert_success_shape(checks, "enum", rc_e, pm_e_json)
        assert_success_shape(checks, "linear", rc_l, pm_l_json)
        checks.check(
            rc_e == 0 and rc_l == 0,
            "both enum and linear runs returned rc=0 (expected for PASS cases)",
            details={"rc_enum": rc_e, "rc_linear": rc_l},
        )
