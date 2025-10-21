# main.py (YAML-driven orchestrator)
# Usage:
#   python main.py --config configs/experiment_1.yaml
# Requirements: PyYAML
#
# This launcher reads a YAML file with a list of steps. Each step contains:
#   - name: str (for logging)
#   - enabled: bool (optional, default True)
#   - cmd: str | list[str]  (command to execute; supports {var} formatting from YAML)
#   - env: dict[str,str]    (optional per-step environment overrides)
#
# Variables available to format inside 'cmd': everything under cfg['paths'] and top-level keys.

# Imports and libraries

import argparse
import subprocess
import sys
import os
import yaml
from datetime import datetime
from pathlib import Path

def run_step(step, cwd=".") -> int:
    """Execute a single pipeline step and return the process return code"""
    if not step.get("enabled", True):
        print(f"[SKIP] {step.get('name','(unnamed)')}")
        return 0

    name = step.get("name", "unnamed")
    cmd  = step.get("cmd")
    env  = os.environ.copy()
    env.update(step.get("env", {}))

    # If cmd is a string we use the shell for convenience
    shell = isinstance(cmd, str)

    print(f"[RUN] {name} -> {cmd}")
    try:
        result = subprocess.run(cmd, cwd=cwd, env=env, shell=shell, check=True)
        print(f"[OK] {name}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {name} (returncode={e.returncode})")
        return e.returncode

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="YAML-driven experiment runner")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML (e.g., configs/experiment_1.yaml)")
    args = parser.parse_args()

    # Read YAML
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name    = cfg.get("experiment_name", "experiment")
    steps       = cfg.get("steps", [])
    logs_dir    = cfg.get("paths", {}).get("logs_dir", "results/logs")
    results_dir = cfg.get("paths", {}).get("results_dir", "results")

    ensure_dirs(logs_dir, results_dir)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"=== [INFO] Starting {exp_name} @ {stamp} ===")

    # Export common environment variables (optional)
    common_env = cfg.get("env", {})
    os.environ.update({k: str(v) for k, v in common_env.items()})

    # Execute steps in order
    for step in steps:
        # Allow string formatting in commands with values from cfg['paths'] and top-level cfg
        cmd = step.get("cmd")
        if isinstance(cmd, str):
            step["cmd"] = cmd.format(**cfg.get("paths", {}), **cfg)
        elif isinstance(cmd, list):
            step["cmd"] = [str(c).format(**cfg.get("paths", {}), **cfg) for c in cmd]

        rc = run_step(step)
        if rc != 0:
            print(f"Pipeline aborted at step: {step.get('name','(unnamed)')}")
            sys.exit(rc)

    print(f"=== [INFO] Finished {exp_name} @ {datetime.now().strftime('%Y%m%d-%H%M%S')} ===")

if __name__ == "__main__":
    main()
