import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pytest

# --------------------------------------------------------------------------------------
# Idea-source marker mapping (objective "ideas" -> plain pytest markers)
# --------------------------------------------------------------------------------------
IDEA_MARKERS = {
    "Capabilities": "capabilities",
    "Failure Modes": "failure_modes",
    "Quality Factors": "quality_factors",
    "Usage Scenarios": "usage_scenarios",
    "Creative Ideas": "creative_ideas",
    # Common aliases / variants:
    "Examples": "examples",
    "Example": "examples",
    "Confirm": "confirm",
}


def _sanitize_nodeid(nodeid: str) -> str:
    """Convert pytest nodeid into a filesystem-friendly name (Windows-safe)."""
    s = nodeid.replace("::", "__")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:180]


# --------------------------------------------------------------------------------------
# Tool selection
# --------------------------------------------------------------------------------------
@pytest.fixture
def tool_path() -> Path:
    """
    Path to the Path_Master script under test.

    Resolution order:
      1) Environment variable PATH_MASTER (absolute or relative to project root)
      2) ../Source_Code/Path_Master_v73.py
      3) ../Source_Code/Path_Master.py
      4) Any Path_Master*.py found in ../Source_Code
    """
    root = Path(__file__).resolve().parents[1]  # Tests/ -> project root

    env = os.environ.get("PATH_MASTER")
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = (root / p).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"PATH_MASTER points to missing file: {p}")

    candidates = [
        root / "Source_Code" / "Path_Master_v73.py",
        root / "Source_Code" / "Path_Master.py",
    ]
    for p in candidates:
        if p.exists():
            return p

    sc = root / "Source_Code"
    if sc.exists():
        for p in sorted(sc.glob("Path_Master*.py")):
            if p.is_file():
                return p

    raise FileNotFoundError(
        "Could not locate Path_Master script. Set PATH_MASTER or place it under Source_Code/."
    )


# --------------------------------------------------------------------------------------
# Artifact directories
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def artifacts_run_dir() -> Path:
    """
    One run directory per pytest invocation.

    Layout:
      <project_root>/Test_Artifacts/<timestamp>_<pid>/
    """
    root = Path(__file__).resolve().parents[1]  # Tests/ -> project root
    out_root = root / "Test_Artifacts"

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pid = os.getpid()
    run_dir = out_root / f"{stamp}_{pid}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@pytest.fixture
def artifacts_dir(request, artifacts_run_dir: Path) -> Path:
    """
    Per-test artifact directory under the session run directory.

    Layout:
      <run_dir>/<sanitized nodeid>/
    """
    test_dir = artifacts_run_dir / _sanitize_nodeid(request.node.nodeid)
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


# --------------------------------------------------------------------------------------
# Checks recorder (per-test PASS/FAIL log + objective metadata)
# --------------------------------------------------------------------------------------
@dataclass
class ChecksRecorder:
    artifacts_dir: Path
    filename: str = "checks.log"
    entries: list[dict] = field(default_factory=list)

    def _write_line(self, line: str) -> None:
        p = self.artifacts_dir / self.filename
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(line)

    def check(
        self,
        condition: bool,
        label: str,
        details: Optional[dict[str, Any]] = None,
        hard: bool = True,
    ) -> bool:
        """Record PASS/FAIL. If hard=True and condition is False, fail the test."""
        entry: dict[str, Any] = {"label": label, "pass": bool(condition)}
        if details is not None:
            entry["details"] = details
        self.entries.append(entry)

        status = "PASS" if condition else "FAIL"
        self._write_line(f"{status}: {label}\n")
        if details is not None:
            self._write_line(f"      details={json.dumps(details, ensure_ascii=False)}\n")

        if hard:
            assert condition, label
        return bool(condition)

    def note(self, label: str, details: Optional[dict[str, Any]] = None) -> None:
        """Record a non-check observation line."""
        entry: dict[str, Any] = {"label": label, "note": True}
        if details is not None:
            entry["details"] = details
        self.entries.append(entry)

        self._write_line(f"NOTE: {label}\n")
        if details is not None:
            self._write_line(f"      details={json.dumps(details, ensure_ascii=False)}\n")


@pytest.fixture
def checks(request, artifacts_dir: Path) -> ChecksRecorder:
    """Per-test recorder that also emits objective metadata into the artifacts folder."""
    rec = ChecksRecorder(artifacts_dir=artifacts_dir)

    m = request.node.get_closest_marker("objective")
    obj = {}
    if m:
        obj = {
            "nodeid": request.node.nodeid,
            "id": m.kwargs.get("id"),
            "text": m.kwargs.get("text"),
            "ideas": m.kwargs.get("ideas", []),
        }

    (artifacts_dir / "objective.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")

    rec._write_line("=== TEST OBJECTIVE ===\n")
    rec._write_line(f"nodeid: {obj.get('nodeid')}\n")
    rec._write_line(f"id: {obj.get('id')}\n")
    rec._write_line(f"text: {obj.get('text')}\n")
    rec._write_line(f"ideas: {obj.get('ideas')}\n")
    rec._write_line("======================\n\n")

    return rec


# --------------------------------------------------------------------------------------
# Collection-time enforcement + objective index
# --------------------------------------------------------------------------------------
def pytest_collection_modifyitems(config, items):
    """
    Enforce: every test must declare exactly one @pytest.mark.objective(...).
    Also emit a note if objective ideas imply markers that are not present.
    """
    missing = []
    multi = []
    mismatches = []

    for item in items:
        obj_markers = list(item.iter_markers(name="objective"))

        if len(obj_markers) == 0:
            missing.append(item.nodeid)
            continue
        if len(obj_markers) > 1:
            multi.append(item.nodeid)
            continue

        m = obj_markers[0]
        ideas = (m.kwargs.get("ideas") or [])
        if not isinstance(ideas, list):
            raise pytest.UsageError(
                f"{item.nodeid}: objective(... ideas=...) must be a list of strings"
            )

        expected_plain = {IDEA_MARKERS[i] for i in ideas if i in IDEA_MARKERS}
        present_plain = {name for name in IDEA_MARKERS.values() if item.get_closest_marker(name)}
        if expected_plain and not expected_plain.issubset(present_plain):
            mismatches.append((item.nodeid, sorted(expected_plain), sorted(present_plain)))

    if missing:
        raise pytest.UsageError(
            "These tests are missing @pytest.mark.objective(id=..., text=..., ideas=[...]):\n"
            + "\n".join("  " + x for x in missing)
        )
    if multi:
        raise pytest.UsageError(
            "These tests have more than one @pytest.mark.objective(...). Use exactly one:\n"
            + "\n".join("  " + x for x in multi)
        )

    if mismatches:
        print("\n[NOTE] Objective idea-sources without matching plain markers (filter tags):")
        for nodeid, expected, present in mismatches[:25]:
            print(f"  {nodeid}")
            print(f"    expected tags: {expected}")
            print(f"    present tags : {present}")


def pytest_collection_finish(session):
    """Write an objectives index file at project root every run."""
    objs = []
    for item in session.items:
        m = item.get_closest_marker("objective")
        if not m:
            continue
        objs.append(
            {
                "nodeid": item.nodeid,
                "objective": {
                    "id": m.kwargs.get("id"),
                    "text": m.kwargs.get("text"),
                    "ideas": m.kwargs.get("ideas", []),
                },
                "idea_tags_present": [
                    name for name in IDEA_MARKERS.values() if item.get_closest_marker(name)
                ],
            }
        )

    out_path = Path(str(session.config.rootpath)) / "objectives_index.json"
    out_path.write_text(json.dumps(objs, indent=2), encoding="utf-8")
