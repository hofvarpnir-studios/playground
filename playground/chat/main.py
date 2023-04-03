import argparse
from playground.chat.trainer import CONFIGS, Trainer
import gin

from playground.tracker.tracker import Tracker


@gin.configurable
def train(
    model_id: str = gin.REQUIRED,
    dataset_path: str = gin.REQUIRED,
    batch_size: int = gin.REQUIRED,
    max_length: int = gin.REQUIRED,
    epochs: int = gin.REQUIRED,
    lr: float = gin.REQUIRED,
    tracker: Tracker = gin.REQUIRED,
    checkpoint_name: str = gin.REQUIRED,
):
    trainer = Trainer(
        model_id=model_id,
        dataset_path=dataset_path,
        batch_size=batch_size,
        max_length=max_length,
        epochs=epochs,
        lr=lr,
        tracker=tracker,
        checkpoint_name=checkpoint_name,
    )
    trainer.train()


def main(config_name: str):
    config_path = CONFIGS / f"{config_name}.gin"
    gin.parse_config_file(config_path)
    train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name", default="basic", type=str, help="Name of the config file"
    )
    args = parser.parse_args()
    main(config_name=args.config_name)
