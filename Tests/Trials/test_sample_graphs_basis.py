import json
import re
import subprocess
from pathlib import Path

import pytest

# Project root: Tests/trials/ -> project root
ROOT = Path(__file__).resolve().parents[2]
TESTDATA = ROOT / "TestData"
TOOLS = ROOT / "Tools"

PATHS_JAR = TOOLS / "paths.jar"

# All sample directed graph inputs (relative to TestData/)
INPUT_FILES = [
    "ECC_001_EBP_001.txt",
    "ECC_002_EBP_006.txt",
    "ECC_003_EBP_006.txt",
    "ECC_004_EBP_007.txt",
    "ECC_005_EBP_004.txt",
    "ECC_006_EBP_005.txt",
    "ECC_007_EBP_006.txt",
    "ECC_008_EBP_001.txt",
    "ECC_009_EBP_001.txt",
    "ECC_010_EBP_007.txt",
    "Example_Input.txt",
    "EX_CF_001_In.txt",
    "EX_CF_002_In.txt",
    "EX_CF_004_In.txt",
    "Funky.txt",
    "inputBMO.txt",
    "LibertyInput.txt",
    "SLG_001_EBP_001.txt",
    "SLG_002_EBP_002.txt",
    "SLG_003_EBP_003.txt",
    "SLG_004_EBP_004.txt",
    "SLG_005_EBP_005.txt",
    "SLG_006_EBP_006.txt",
    "SLG_007_EBP_007.txt",
    "SLG_008_EBP_008.txt",
    "SLG_009_EBP_009.txt",
    "SLG_010_EBP_010.txt",
    "T005.txt",
    "T010.txt",
    "T015.txt",
    "T016.txt",
    "T017.txt",
    "T018.txt",
    "T020.txt",
    "T050.txt",
]


def run_paths_jar(cfg_file: Path, timeout_s: int = 15) -> tuple[int, str, str]:
    # Baseline oracle: java -jar paths.jar ..\TestData\X.txt -basis
    cmd = ["java", "-jar", str(PATHS_JAR), str(cfg_file), "-basis"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
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


def run_path_master_json(tool_path: Path, cfg_file: Path, extra_args=None, timeout_s: int = 5) -> tuple[int, dict, str, str]:
    extra_args = extra_args or []
    cmd = ["python", str(tool_path), "--json", "--confirm-basis", *extra_args, str(cfg_file)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
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
    id="OBJ-GRAPH-001",
    text="Confirm basis paths can be generated with a confirmed set of basis paths and the same number of basis paths as paths.jar, running Path_Master.py with both linear and enumerated methods. Timeout of 5 seconds is imposed on all calls to Path_Master.py.",
    ideas=["Capabilities", "Examples"],
)
@pytest.mark.parametrize("input_name", INPUT_FILES)
def test_basis_generation_against_paths_jar_on_samples(tool_path, artifacts_dir, checks, input_name):
    cfg_file = TESTDATA / input_name

    # Preconditions
    checks.check(cfg_file.exists(), f"Precondition: input exists: {cfg_file}")
    checks.check(PATHS_JAR.exists(), f"Precondition: paths.jar exists: {PATHS_JAR}")

    # Put each input's artifacts in its own subfolder for readability
    case_dir = artifacts_dir / cfg_file.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    checks.note("Case directory", details={"case_dir": str(case_dir)})

    # --- Baseline: paths.jar (counts only; do NOT compare path lists) ---
    try:
        rc_j, out_j, err_j = run_paths_jar(cfg_file)
    except subprocess.TimeoutExpired:
        checks.check(False, "paths.jar completed within timeout", details={"timeout_s": 15})
        return

    (case_dir / "paths_jar_stdout.txt").write_text(out_j, encoding="utf-8")
    (case_dir / "paths_jar_stderr.txt").write_text(err_j, encoding="utf-8")

    checks.check(rc_j == 0, "paths.jar exits with rc=0", details={"rc": rc_j})
    base = parse_paths_jar_counts(out_j)
    checks.note("Baseline counts from paths.jar", details=base)

    # --- Path_Master enumeration (default), with 5s timeout ---
    try:
        rc_e, pm_e_json, pm_e_stdout, pm_e_stderr = run_path_master_json(Path(tool_path), cfg_file, extra_args=[], timeout_s=5)
    except subprocess.TimeoutExpired:
        checks.check(False, "Path_Master(enum) completed within timeout", details={"timeout_s": 5})
        return

    (case_dir / "path_master_enum.json").write_text(json.dumps(pm_e_json, indent=2), encoding="utf-8")
    (case_dir / "path_master_enum_stdout.json").write_text(pm_e_stdout + "\n", encoding="utf-8")
    (case_dir / "path_master_enum_stderr.txt").write_text(pm_e_stderr, encoding="utf-8")

    checks.check(rc_e == 0, "Path_Master(enum) exits with rc=0", details={"rc": rc_e})
    enum_counts = extract_pm_counts(pm_e_json)
    checks.note("Path_Master(enum) extracted counts", details=enum_counts)

    # --- Path_Master linear, with 5s timeout ---
    try:
        rc_l, pm_l_json, pm_l_stdout, pm_l_stderr = run_path_master_json(Path(tool_path), cfg_file, extra_args=["--linear"], timeout_s=5)
    except subprocess.TimeoutExpired:
        checks.check(False, "Path_Master(linear) completed within timeout", details={"timeout_s": 5})
        return

    (case_dir / "path_master_linear.json").write_text(json.dumps(pm_l_json, indent=2), encoding="utf-8")
    (case_dir / "path_master_linear_stdout.json").write_text(pm_l_stdout + "\n", encoding="utf-8")
    (case_dir / "path_master_linear_stderr.txt").write_text(pm_l_stderr, encoding="utf-8")

    checks.check(rc_l == 0, "Path_Master(linear) exits with rc=0", details={"rc": rc_l})
    lin_counts = extract_pm_counts(pm_l_json)
    checks.note("Path_Master(linear) extracted counts", details=lin_counts)

    # --- Checks: each variant self-consistency + compare counts to baseline ---
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

        # Compare counts vs baseline
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
