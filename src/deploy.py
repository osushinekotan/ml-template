import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(search_from="./", indicator=".project-root", pythonpath=True)

from kaggle import KaggleApi

from src.kaggle.datasets import Deploy


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Push experiment outputs to kaggle dataset."""
    client = KaggleApi()
    client.authenticate()

    deploy = Deploy(cfg=cfg, client=client)

    # deploy trained model weights
    deploy.push_output(
        ignore_patterns=[
            ".git",
            "__pycache__",
            "checkpoints",
            "*.log",
            "csv",
            "wandb",
            "*.pkl",
            "*.csv",
            ".ipynb_checkpoints",
        ]
    )

    deploy.push_code()

    hf_model_name = cfg.get("hf_model_name")  # if None, it will not push the hugguingface pretrained model
    deploy.push_huguingface_model(model_name=hf_model_name)


if __name__ == "__main__":
    main()
