"""The handler for training and evaluation."""

import os
import sys
from argparse import Namespace

import pytest
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from difusco.arg_parser import get_arg_parser
from difusco.tsp.pl_tsp_model import TSPModel

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Lightning version: {pl.__version__}")


@pytest.mark.skipif(sys.version_info >= (3, 10), reason="Checkpoints are supported in old versions of PyTorch")
def test_loading_checkpoint() -> None:
    arg_parser = get_arg_parser()
    args = Namespace(**{action.dest: action.default for action in arg_parser._actions})  # noqa: SLF001
    args.data_path = "data"
    args.models_path = "models"
    args.training_split = "tsp/tsp50_train_concorde.txt"
    args.validation_split = "tsp/tsp50_test_concorde.txt"
    args.test_split = "tsp/tsp50_test_concorde.txt"
    args.diffusion_type = "categorical"
    args.learning_rate = 0.0002
    args.weight_decay = 0.0001
    args.lr_scheduler = "cosine-decay"
    args.batch_size = 32
    args.num_epochs = 25
    args.inference_schedule = "cosine"
    args.inference_diffusion_steps = 50
    args.resume_weight_only = True

    lr_callback = LearningRateMonitor(logging_interval="step")

    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
    wandb_logger = WandbLogger(
        name="tsp_diffusion_graph_categorical_tsp50_test",
        project="difusco_tsp",
        entity=None,
        save_dir=os.path.join("models", "tsp"),
        id=wandb_id,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/solved_cost",
        mode="min",
        save_top_k=3,
        save_last=True,
        dirpath=os.path.join(
            wandb_logger.save_dir,
            "tsp_diffusion_graph_categorical_tsp50_test",
            wandb_logger._id,  # noqa: SLF001
            "checkpoints",
        ),
    )

    trainer = Trainer(
        accelerator="auto",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        max_epochs=50,
        callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback],
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        strategy=DDPStrategy(static_graph=True),
        precision=32,
    )

    ckpt_path = "/home/e12223411/repos/difusco/models/tsp/tsp50_categorical.ckpt"
    model = TSPModel(param_args=args)

    trainer.validate(model, ckpt_path=ckpt_path)
    trainer.test(model, ckpt_path=ckpt_path)


if __name__ == "__main__":
    test_loading_checkpoint()
