import ast
import json
from pathlib import Path

import chi_error_nbar_workflow as workflow


ROOT = Path(__file__).parent


def test_default_config_disables_expensive_recomputation():
    config = workflow.default_config()
    assert config["RECOMPUTE"] is False
    assert config["RUN_EXACT_FULL_CHI_SWEEP"] is False
    assert config["RUN_NUMERICAL_CONVERGENCE"] is False
    assert config["RUN_NOISE_SOURCE_ABLATION"] is False
    assert config["RUN_HXX_DRIVE_CALIBRATION_QPT"] is False
    assert config["RUN_PHYSICAL_CONTROL_QPT"] is False
    assert config["RUN_PARAMETER_ROBUSTNESS_QPT"] is False


def test_run_merges_partial_simulation_overrides(monkeypatch):
    captured = {}

    def fake_run_path(path, init_globals):
        captured["path"] = path
        captured["config"] = init_globals["WORKFLOW_CONFIG"]
        return {"ok": True}

    monkeypatch.setattr(workflow.runpy, "run_path", fake_run_path)
    result = workflow.run({
        "FIT_DEGREE": 3,
        "SIMULATION_PARAMS": {"A": 0.2},
    })

    assert result == {"ok": True}
    assert captured["config"]["FIT_DEGREE"] == 3
    assert captured["config"]["SIMULATION_PARAMS"]["A"] == 0.2
    assert captured["config"]["SIMULATION_PARAMS"]["eta"] == 0.1


def test_notebook_contains_configuration_calls_but_no_function_definitions():
    notebook = json.loads(
        (ROOT / "chi_error_element_nbar_fit.ipynb").read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    tree = ast.parse(code)
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )
    assert "workflow.run(CONFIG)" in code
    cell_ids = {cell.get("id") for cell in notebook["cells"]}
    assert {
        "independent-drive-qpt",
        "independent-fock-xx-angle",
        "independent-kirchhoff",
        "independent-control",
        "independent-robustness",
    } <= cell_ids
    assert (
        'DRIVE_RE_QPT_NBARS = list(CONFIG["HXX_DRIVE_CALIBRATION_NBARS"])'
        in code
    )
    assert "CONTROL_QPT_NBARS =" in code
    assert "ROBUSTNESS_QPT_NBARS =" in code
    assert "SHOW_CONTROL_PROGRESS = True" in code
    assert "SHOW_ROBUSTNESS_PROGRESS = True" in code
