import json
import re
import subprocess
from pathlib import Path

import pytest

# Project root: Tests/trials/ -> project root
ROOT = Path(__file__).resolve().parents[2]
TESTDATA = ROOT / "TestData"
TOOLS = ROOT / "Tools"

EX_CF_004_INPUT = TESTDATA / "EX_CF_004_In.txt"
PATHS_JAR = TOOLS / "paths.jar"


def run_paths_jar(ex_cf_004_input: Path) -> tuple[int, str, str]:
    cmd = ["java", "-jar", str(PATHS_JAR), str(ex_cf_004_input), "-basis"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return p.returncode, (p.stdout or ""), (p.stderr or "")


def parse_paths_jar_counts(stdout_text: str) -> dict:
    # Expect lines like:
    #   Cyclomatic Complexity: 54
    #   Number of basis paths: 54
    cyclo_m = re.search(r"Cyclomatic\s+Complexity:\s*(\d+)", stdout_text, flags=re.IGNORECASE)
    basis_m = re.search(r"Number\s+of\s+basis\s+paths:\s*(\d+)", stdout_text, flags=re.IGNORECASE)

    if not cyclo_m or not basis_m:
        raise AssertionError(
            "Could not parse required counts from paths.jar output.\n"
            "Expected lines like 'Cyclomatic Complexity: <n>' and 'Number of basis paths: <n>'."
        )

    return {
        "cyclomatic_complexity": int(cyclo_m.group(1)),
        "basis_count": int(basis_m.group(1)),
    }


def run_path_master_json(tool_path: Path, ex_cf_004_input: Path, extra_args=None) -> tuple[int, dict, str, str]:
    extra_args = extra_args or []
    cmd = ["python", str(tool_path), "--json", "--confirm-basis", *extra_args, str(ex_cf_004_input)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    stdout = (p.stdout or "").strip()
    if not stdout:
        raise AssertionError(f"Path_Master produced empty stdout.\nSTDERR:\n{p.stderr}")

    obj = json.loads(stdout)
    return p.returncode, obj, stdout, (p.stderr or "")


def _find_key_anywhere(obj, key: str):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key_anywhere(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for it in obj:
            found = _find_key_anywhere(it, key)
            if found is not None:
                return found
    return None


def extract_pm_counts(pm_json: dict) -> dict:
    """
    Extract counts from Path_Master JSON output.

    Canonical (v73+ observed):
      results.analysis.relevant_subgraph.cyclomatic_complexity
      results.analysis.basis.basis_count
      results.analysis.confirm_basis.pass

    We also include a fallback search for resilience across minor schema reshapes.
    """
    analysis = (pm_json.get("results") or {}).get("analysis") or {}
    sub = analysis.get("relevant_subgraph") or {}

    cyclo = sub.get("cyclomatic_complexity")
    if cyclo is None:
        cyclo = analysis.get("cyclomatic_complexity")
    if cyclo is None:
        cyclo = _find_key_anywhere(pm_json, "cyclomatic_complexity")

    basis = (analysis.get("basis") or {}).get("basis_count")
    if basis is None:
        basis = _find_key_anywhere(pm_json, "basis_count")

    confirm_pass = (analysis.get("confirm_basis") or {}).get("pass")
    if confirm_pass is None:
        confirm_pass = _find_key_anywhere(pm_json, "pass")

    return {"cyclomatic_complexity": cyclo, "basis_count": basis, "confirm_pass": confirm_pass}


@pytest.mark.capabilities
@pytest.mark.examples
@pytest.mark.objective(
    id="OBJ-LIB-002",
    text="Confirm Path_Master.py can generate a set of basis paths for ECSE 429 class exercise EX_CF_004. For comparison purposes a baseline set of basis paths is generated with the java program paths.jar.",
    ideas=["Capabilities", "Examples"],
)
def test_ex_cf_004_basis_enum_and_linear(tool_path, artifacts_dir, checks):
    # Preconditions
    checks.check(EX_CF_004_INPUT.exists(), f"Precondition: ex_cf_004 input exists: {EX_CF_004_INPUT}")
    checks.check(PATHS_JAR.exists(), f"Precondition: paths.jar exists: {PATHS_JAR}")

    # --- Baseline: paths.jar (counts only; do NOT compare paths) ---
    rc_j, out_j, err_j = run_paths_jar(EX_CF_004_INPUT)
    (artifacts_dir / "paths_jar_stdout.txt").write_text(out_j, encoding="utf-8")
    (artifacts_dir / "paths_jar_stderr.txt").write_text(err_j, encoding="utf-8")

    checks.note(
        "Saved paths.jar stdout/stderr",
        details={
            "stdout": str(artifacts_dir / "paths_jar_stdout.txt"),
            "stderr": str(artifacts_dir / "paths_jar_stderr.txt"),
        },
    )

    checks.check(
        rc_j == 0,
        "paths.jar exits with rc=0",
        details={"rc": rc_j},
    )

    base = parse_paths_jar_counts(out_j)
    checks.note("Baseline counts from paths.jar (counts only; paths not compared)", details=base)

    # --- Path_Master enumeration (default) ---
    rc_e, pm_e_json, pm_e_stdout, pm_e_stderr = run_path_master_json(Path(tool_path), EX_CF_004_INPUT, extra_args=[])
    (artifacts_dir / "path_master_enum.json").write_text(json.dumps(pm_e_json, indent=2), encoding="utf-8")
    (artifacts_dir / "path_master_enum_stdout.json").write_text(pm_e_stdout + "\n", encoding="utf-8")
    (artifacts_dir / "path_master_enum_stderr.txt").write_text(pm_e_stderr, encoding="utf-8")

    checks.note(
        "Saved Path_Master(enum) artifacts",
        details={
            "json": str(artifacts_dir / "path_master_enum.json"),
            "stdout_json": str(artifacts_dir / "path_master_enum_stdout.json"),
            "stderr": str(artifacts_dir / "path_master_enum_stderr.txt"),
        },
    )

    checks.check(
        rc_e == 0,
        "Path_Master(enum) exits with rc=0",
        details={"rc": rc_e},
    )

    enum_counts = extract_pm_counts(pm_e_json)
    checks.note("Path_Master(enum) extracted counts", details=enum_counts)

    # --- Path_Master linear ---
    rc_l, pm_l_json, pm_l_stdout, pm_l_stderr = run_path_master_json(Path(tool_path), EX_CF_004_INPUT, extra_args=["--linear"])
    (artifacts_dir / "path_master_linear.json").write_text(json.dumps(pm_l_json, indent=2), encoding="utf-8")
    (artifacts_dir / "path_master_linear_stdout.json").write_text(pm_l_stdout + "\n", encoding="utf-8")
    (artifacts_dir / "path_master_linear_stderr.txt").write_text(pm_l_stderr, encoding="utf-8")

    checks.note(
        "Saved Path_Master(linear) artifacts",
        details={
            "json": str(artifacts_dir / "path_master_linear.json"),
            "stdout_json": str(artifacts_dir / "path_master_linear_stdout.json"),
            "stderr": str(artifacts_dir / "path_master_linear_stderr.txt"),
        },
    )

    checks.check(
        rc_l == 0,
        "Path_Master(linear) exits with rc=0",
        details={"rc": rc_l},
    )

    lin_counts = extract_pm_counts(pm_l_json)
    checks.note("Path_Master(linear) extracted counts", details=lin_counts)

    # --- Assertions / learning checks ---
    for label, got in [("enum", enum_counts), ("linear", lin_counts)]:
        checks.check(got["cyclomatic_complexity"] is not None, f"{label}: cyclomatic_complexity present")
        checks.check(got["basis_count"] is not None, f"{label}: basis_count present")

        checks.check(
            got["basis_count"] == got["cyclomatic_complexity"],
            f"{label}: basis_count == cyclomatic_complexity (Path_Master invariant)",
            details={"basis_count": got["basis_count"], "cyclomatic_complexity": got["cyclomatic_complexity"]},
        )

        checks.check(
            got["confirm_pass"] is True,
            f"{label}: --confirm-basis passes",
            details={"confirm_pass": got["confirm_pass"]},
        )

        # Compare counts vs baseline (counts only; NOT paths)
        checks.check(
            int(got["cyclomatic_complexity"]) == int(base["cyclomatic_complexity"]),
            f"{label}: cyclomatic complexity matches paths.jar baseline",
            details={"path_master": got["cyclomatic_complexity"], "paths_jar": base["cyclomatic_complexity"]},
        )

        checks.check(
            int(got["basis_count"]) == int(base["basis_count"]),
            f"{label}: basis_count matches paths.jar baseline",
            details={"path_master": got["basis_count"], "paths_jar": base["basis_count"]},
        )

    # Compare Path_Master variants to each other (counts only)
    checks.check(
        int(enum_counts["cyclomatic_complexity"]) == int(lin_counts["cyclomatic_complexity"]),
        "enum vs linear: cyclomatic complexity matches",
        details={"enum": enum_counts["cyclomatic_complexity"], "linear": lin_counts["cyclomatic_complexity"]},
    )
    checks.check(
        int(enum_counts["basis_count"]) == int(lin_counts["basis_count"]),
        "enum vs linear: basis_count matches",
        details={"enum": enum_counts["basis_count"], "linear": lin_counts["basis_count"]},
    )
