# type: ignore
def run(
    experiment: str,
    code: str,
    alias: str,
) -> None:
    """Run inference notebook in Kaggle.

    Args:
    ----
        experiment (str): experiment name.
        code (str, optional): input dir in kaggle kernel. e.g. "/kaggle/input/{competition-name}"
        alias (str, optional): alias name for the experiment. e.g. "{competition-alias}"

    """
    import os
    import subprocess

    os.chdir(code)
    os.environ["PYTHONPATH"] = code
    os.environ["ENVIRONMENT"] = "kaggle"

    output_dir = f"/kaggle/input/{alias}-experiment-{experiment}"
    command = [
        "python",
        "src/inference.py",
        f"--config-path={code}/configs",
        "paths=kaggle",
        "hydra=kaggle",
        f"experiment={experiment}",
        f"paths.output_dir={output_dir}",
        f"paths.code_dir={code}",
    ]

    subprocess.run(command)
