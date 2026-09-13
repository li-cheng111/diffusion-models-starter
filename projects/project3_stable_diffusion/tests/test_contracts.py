import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_parse():
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_manual_notebook_does_not_call_high_level_pipeline():
    notebook = json.loads((ROOT / "01_inference_walkthrough.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "StableDiffusionPipeline" not in code
    assert "pipe(" not in code
    assert "scheduler.step" in code
    assert "vae.decode" in code


def test_project3_notebooks_are_valid_json():
    for path in ROOT.glob("*.ipynb"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert any(cell["cell_type"] == "code" for cell in payload["cells"])
