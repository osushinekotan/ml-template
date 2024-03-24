import logging
import os
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ENVIRONMENT = os.environ.get("ENVIRONMENT", "local")
RUN_MODE = os.environ.get("RUN_MODE", "experiment")


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run inference notebook."""
    env = os.environ.copy()
    env["OVERRIDES"] = cfg.experiment_name  # NOTE:  All Overrides not just experiment_name

    # check run mode
    if RUN_MODE in ["experiment", "inference"]:
        raise ValueError(f"Invalid run mode: {RUN_MODE}")

    run_dir = Path(cfg.paths.inferences_dir) if RUN_MODE == "inference" else Path(cfg.paths.experiments_dir)

    complied_notebook_name = f"compiled_{cfg.notebook}_{RUN_MODE}.ipynb"
    if ENVIRONMENT == "kaggle":
        logger.info("Running on Kaggle")
        compiled_notebook_path = Path(cfg.paths.root_dir) / complied_notebook_name
    elif ENVIRONMENT == "local":
        logger.info("Running on local")
        compiled_notebook_path = Path(cfg.paths.output_dir) / complied_notebook_name
    else:
        raise ValueError(f"Unknown environment: {ENVIRONMENT}")

    target_notebook = (run_dir / f"{cfg.notebook}.ipynb").as_posix()
    logger.info(f"Overrides: {env['OVERRIDES']}, Notebook: {target_notebook}")

    command = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--output",
        compiled_notebook_path.as_posix(),
        "--ExecutePreprocessor.timeout=None",
        "--ExecutePreprocessor.allow_errors=True",
        target_notebook,
    ]  # Unsure how to set environment variables using nbconvert.ExecutePreprocessor

    try:
        subprocess.run(command, env=env, check=True)
        logger.info("Finished running notebook successfully! 🎉")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run notebook 😭: {e}")


if __name__ == "__main__":
    main()
