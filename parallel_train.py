import itertools  # Used to generate hyperparameter combinations
import os
import subprocess
import sys
from multiprocessing import Pool

import psutil

# Global settings
TARGET_SCRIPT = os.path.join(os.path.dirname(__file__), "train_model.py")
LOGS_DIR = "logs"
N = 1
USE_MP = False
HP_OPTIONS = {
    "optimizer": ["v0", "v1", "v2"],
    "freeze": [True, False],
    "contraster_type": ["manual", "pil", "torch", None],
}


def generate_hyperparam_combinations():
    """Generate all possible hyperparameter combinations as command-line arguments."""
    keys = HP_OPTIONS.keys()
    values = HP_OPTIONS.values()

    # Get all possible combinations of hyperparameters
    for combination in itertools.product(*values):
        # Convert to command-line format:
        cmd_args = [f"--{key} {value}" for key, value in zip(keys, combination)]
        yield cmd_args


def run_script_with_params(params, mp=False):
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    log_file = f"{LOGS_DIR}/log_{params.replace('--', '_').replace(' ', '_')}.log"
    command = f"python {TARGET_SCRIPT} {params}"

    with open(log_file, "w") as log:
        if mp:
            subprocess.Popen(command, stdout=log, stderr=log)
        else:
            print(f"Running command: {command}")
            subprocess.run(command, stdout=log, stderr=log)
            print(f"Command finished: {command}")


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

    global N, USE_MP
    if USE_MP:
        with Pool(processes=N) as pool:
            pool.starmap(
                run_script_with_params,
                [(" ".join(params),) for params in param_combinations],
            )
        print("All processes started.")
    else:
        for params in param_combinations:
            run_script_with_params(" ".join(params))
        print("All processes started.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "kill":
        kill_existing_processes()
    else:
        main()
