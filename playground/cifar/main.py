import argparse
from playground.cifar.trainer import CIFARData, Trainer
import gin
import torch.nn as nn

from playground.cifar.utils import CONFIGS
from playground.tracker.tracker import Tracker


@gin.configurable
def train_and_evaluate(
    data: CIFARData = gin.REQUIRED,
    network: nn.Module = gin.REQUIRED,
    epochs: int = gin.REQUIRED,
    lr: float = gin.REQUIRED,
    tracker: Tracker = gin.REQUIRED,
):
    trainer = Trainer(epochs, data, network, lr, tracker)
    trainer.train()
    trainer.validate_by_class()


def main(config_name: str):
    config_path = CONFIGS / f"{config_name}.gin"
    gin.parse_config_file(config_path)
    train_and_evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name", default="mlp", type=str, help="Name of the config file"
    )
    args = parser.parse_args()
    main(config_name=args.config_name)
