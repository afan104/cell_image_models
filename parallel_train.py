import itertools  # Used to generate hyperparameter combinations
import os
import subprocess
import sys
from multiprocessing import Pool

import psutil
import torch

from utils.dataset_manipulation import train_tfms1, train_tfms2
from utils.model_classes import AdaptiveContrastPreprocessing

# Global settings
VENV_ACTIVATE = os.path.join(os.getcwd(), ".venv", "Scripts", "activate")
VENV_PYTHON = os.path.join(os.getcwd(), ".venv", "Scripts", "python")
TARGET_SCRIPT = "train_model.py"
OUTPUT_DIR = "logs"


OPS_HPS = {
    "v0": {"optimizer": torch.optim.SGD, "lr": 0.2},
    "v1": {
        "optimizer": torch.optim.SGD,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 3e-4,
    },
    "v2": {
        "optimizer": torch.optim.Adam,
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "weight_decay": 1e-3,
    },
}

hyperparam_options = {
    "optimizer": ["v0", "v1", "v2"],
    "data_augmentation": ["v0", "v1"],
    "freeze": [True, False],
    "preprocess": ["v0", "v1"],
}


def generate_hyperparam_combinations():
    """Generate all possible hyperparameter combinations as command-line arguments."""
    optimizer_values = hyperparam_options["optimizer"]  # ["v0", "v1", "v2"]
    other_keys = [k for k in hyperparam_options if k != "optimizer"]
    other_values = [v for k, v in hyperparam_options.items() if k != "optimizer"]

    special_cases = {
        "freeze": True,
        "preprocess": "v1",
        "data_augmentation": "v1",
    }

    # Get all possible combinations of hyperparameters
    for combination in itertools.product(*other_values):
        # Convert to command-line format:
        cmd_args = [f"--{key} {value}" for key, value in zip(other_keys, combination)]

        # yield with base optimizer\
        yield cmd_args + [f"--optimizer {optimizer_values[0]}"]

        # check special case criteria
        is_special_case = any(
            key in special_cases and value == special_cases[key]
            for key, value in zip(other_keys, combination)
        )
        if is_special_case:
            # yields two additional versions
            yield cmd_args + [f"--optimizer {optimizer_values[1]}"]
            yield cmd_args + [f"--optimizer {optimizer_values[2]}"]


def run_script_with_params(params):
    """Runs the target script with the given parameters."""
    print(f"params: {params}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    log_file = f"{OUTPUT_DIR}/log_{params.replace('--', '_').replace(' ', '_')}.log"
    command = f"python {TARGET_SCRIPT} {params}"
    print(f"Running command: {command}")

    with open(log_file, "w") as log:
        subprocess.Popen(command, stdout=log, stderr=log)


def kill_existing_processes():
    """Kills all processes from previous runs."""
    print("Killing existing processes...")
    for proc in psutil.process_iter():
        try:
            if TARGET_SCRIPT in proc.cmdline():
                print(f"Killing process {proc.pid}...")
                proc.terminate()
                proc.wait(timeout=3)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    print("All existing processes terminated.")


def main():
    """Main function to set up parallel execution."""
    kill_existing_processes()
    param_combinations = list(generate_hyperparam_combinations())
    with Pool(processes=3) as pool:
        pool.starmap(
            run_script_with_params,
            [(" ".join(params),) for params in param_combinations],
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "kill":
        kill_existing_processes()
    else:
        main()
