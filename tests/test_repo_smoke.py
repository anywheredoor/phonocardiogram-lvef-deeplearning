from __future__ import annotations

import json
import py_compile
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip()
        if name:
            names.add(name.lower())
    return names


def test_python_sources_compile() -> None:
    python_files = list((REPO_ROOT / "src").rglob("*.py")) + list(
        (REPO_ROOT / "scripts").rglob("*.py")
    )
    assert python_files, "Expected repository Python files to exist."

    for path in python_files:
        py_compile.compile(str(path), doraise=True)


def test_notebook_code_cells_compile() -> None:
    notebook_path = REPO_ROOT / "colab_pipeline.ipynb"
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    compiled_cells = 0
    for index, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        cleaned = "\n".join(
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("!", "%"))
        )
        if not cleaned.strip():
            continue
        compile(cleaned, f"{notebook_path.name} cell {index}", "exec")
        compiled_cells += 1

    assert compiled_cells > 0


def test_requirements_cover_runtime_dependencies() -> None:
    required = {
        "torch",
        "torchaudio",
        "timm",
        "pandas",
        "numpy",
        "scikit-learn",
        "soundfile",
        "tqdm",
        "gammatone",
        "matplotlib",
        "seaborn",
    }

    requirements = _requirement_names(REPO_ROOT / "requirements.txt")
    locked_requirements = _requirement_names(REPO_ROOT / "requirements-lock.txt")

    assert required.issubset(requirements)
    assert required.issubset(locked_requirements)
