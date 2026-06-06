import os
import json
import subprocess
import datetime
from pathlib import Path

RELDDPM_DIR = Path(__file__).resolve().parents[1]

def _run(argv: list[str]) -> None:
    import sys
    cmd = [sys.executable, *argv]
    proc = subprocess.run(cmd, cwd=str(RELDDPM_DIR), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{argv[0]} exited {proc.returncode}: {proc.stderr.strip()[-500:]}")

def train(req: dict) -> Path:
    abs_model_dir = os.path.abspath(req["output_model_dir"])
    os.makedirs(abs_model_dir, exist_ok=True)
    dataset = req["dataset"]
    train_csv = os.path.abspath(req["train_csv"])
    label_col = req.get("label_column")

    # Train Diffuser
    _run([
        "main.py", 
        f"--task-name=train_diffuser", 
        f"--dataset-name={dataset}",
        f"--train-csv={train_csv}",
        f"--label-col={label_col}",
        f"--save-name={abs_model_dir}"
    ])

    # Train Controller (since original RelDDPM uses one controller for the label column)
    _run([
        "main.py", 
        f"--task-name=train_controller", 
        f"--dataset-name={dataset}",
        f"--train-csv={train_csv}",
        f"--label-col={label_col}",
        f"--save-name={abs_model_dir}"
    ])

    meta = {
        "method": "RelDDPM",
        "dataset": dataset,
        "label_column": label_col,
        "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    Path(abs_model_dir, "meta.json").write_text(json.dumps(meta, indent=2))
    return Path(abs_model_dir)

def generate(req: dict, native_constraints: list[str]) -> Path:
    abs_model_dir = os.path.abspath(req["model_dir"])
    meta = json.loads(Path(abs_model_dir, "meta.json").read_text())
    
    dataset = meta["dataset"]
    label_col = meta.get("label_column")
    n_samples = int(req["n_samples"])
    output_csv = os.path.abspath(req["output_csv_path"])
    
    if len(native_constraints) != 1:
        raise ValueError("Original RelDDPM only supports exactly one constraint.")
        
    c = native_constraints[0]
    col, val = c.split("=", 1)
    if col != label_col:
        raise ValueError(f"Constraint column '{col}' does not match trained label column '{label_col}'.")

    _run([
        "main.py", 
        f"--task-name=sample", 
        f"--dataset-name={dataset}",
        f"--save-name={abs_model_dir}",
        f"--target-class={val}",
        f"--n-samples={n_samples}",
        f"--output-csv={output_csv}"
    ])

    return Path(output_csv)
