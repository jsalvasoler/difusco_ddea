"""The handler for training and evaluation."""

import os
from argparse import Namespace

import torch
from pyinstrument import Profiler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

import wandb
from difusco.arg_parser import parse_args
from difusco.mis.pl_high_degree_model import HighDegreeSelection
from difusco.mis.pl_mis_model import MISModel
from difusco.tsp.pl_tsp_model import TSPModel


def difusco(args: Namespace) -> None:
    epochs = args.num_epochs
    project_name = args.project_name

    if args.task == "tsp":
        model_class = TSPModel
        saving_mode = "min"
        monitor = "val/solved_cost"
    elif args.task == "mis":
        model_class = MISModel
        saving_mode = "max"
        monitor = "val/solved_cost"
    elif args.task == "high_degree_selection":
        model_class = HighDegreeSelection
        saving_mode = "min"
        monitor = "val/accuracy"
    else:
        raise NotImplementedError

    model = model_class(param_args=args)

    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
    print(args.wandb_logger_name)
    wandb_logger = WandbLogger(
        name=args.wandb_logger_name,
        project=project_name,
        entity=args.wandb_entity,
        save_dir=args.logs_path,
        id=args.resume_id or wandb_id,
    )
    rank_zero_info(f"Logging to {wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}")

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode=saving_mode,
        save_top_k=3,
        save_last=True,
        dirpath=os.path.join(
            wandb_logger.save_dir,
            args.wandb_logger_name,
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
        precision=16 if args.fp16 else 32,
    )

    rank_zero_info(f"{'-' * 100}\n" f"{model.model!s}\n" f"{'-' * 100}\n")

    ckpt_path = os.path.join(args.models_path, args.ckpt_path) if args.ckpt_path else None

    if args.do_train:
        if args.resume_weight_only:
            model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
            trainer.fit(model)
        else:
            trainer.fit(model, ckpt_path=ckpt_path)

        if args.do_test:
            trainer.test(ckpt_path=checkpoint_callback.best_model_path)

    elif args.do_test:
        trainer.validate(model, ckpt_path=ckpt_path)
        if not args.do_valid_only:
            trainer.test(model, ckpt_path=ckpt_path)
    trainer.logger.finalize("success")


def main_difusco(args: Namespace) -> None:
    if args.profiler:
        with Profiler() as p:
            difusco(args)
        print(p.output_text(unicode=True, color=True))
    else:
        difusco(args)


if __name__ == "__main__":
    args = parse_args()
    main_difusco(args)
