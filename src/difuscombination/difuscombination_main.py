"""The handler for training and evaluation."""

import os

import torch
from config.myconfig import Config
from pyinstrument import Profiler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

import wandb
from difuscombination.pl_difuscombination_mis_model import DifusCombinationMISModel


def difuscombination(config: Config) -> None:
    epochs = config.num_epochs
    project_name = config.project_name

    if config.task == "mis":
        model_class = DifusCombinationMISModel
        saving_mode = "max"
        monitor = "val/solved_cost"
    elif config.task == "tsp":
        raise NotImplementedError("DifuscoCombination is not yet implemented for TSP")
    else:
        raise NotImplementedError

    model = model_class(config=config)

    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
    print(config.wandb_logger_name)
    wandb_logger = WandbLogger(
        name=config.wandb_logger_name,
        project=project_name,
        entity=config.wandb_entity,
        save_dir=config.logs_path,
        id=config.resume_id or wandb_id,
    )
    rank_zero_info(
        f"Logging to {wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode=saving_mode,
        save_top_k=3,
        save_last=True,
        dirpath=os.path.join(
            wandb_logger.save_dir,
            config.wandb_logger_name,
            wandb_logger._id,  # noqa: SLF001
            "checkpoints",
        ),
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 4,
        max_epochs=epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback],
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        strategy=DDPStrategy(static_graph=True),
        precision=16 if config.fp16 else 32,
    )

    rank_zero_info(f"{'-' * 100}\n{model.model!s}\n{'-' * 100}\n")

    ckpt_path = (
        os.path.join(config.models_path, config.ckpt_path) if config.ckpt_path else None
    )

    if config.do_train:
        if config.resume_weight_only:
            model = model_class.load_from_checkpoint(ckpt_path, config=config)
            trainer.fit(model)
        else:
            trainer.fit(model, ckpt_path=ckpt_path)

        if config.do_test:
            trainer.test(ckpt_path=checkpoint_callback.best_model_path)

    elif config.do_test:
        trainer.validate(model, ckpt_path=ckpt_path)
        if not config.do_valid_only:
            trainer.test(model, ckpt_path=ckpt_path)
    trainer.logger.finalize("success")


def main_difuscombination(config: Config) -> None:
    if config.profiler:
        with Profiler() as p:
            difuscombination(config)
        print(p.output_text(unicode=True, color=True))
    else:
        difuscombination(config)
