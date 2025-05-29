import hydra
import lightning.pytorch as pl
import pandas as pd
import torchmetrics
import wandb

from pathlib import Path
import sys
from classes import ResnetClassifier

from models import (
    ResNet18, 
    ResNet18ToFinetune
)
from data import HFImageDataset

import sys
curr_path = Path(__file__)
sys.path.append(str(curr_path.parent.parent / "ALBERT_fine_tuning/functions"))
from reproducibility import set_seed

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
import torch
from torch import nn
from torch.utils.data import DataLoader
import os

@hydra.main(config_path="configs", config_name=None, version_base="1.3")
def main(cfg: DictConfig):
    torch.manual_seed(42)

    WANDB_PROJECT_NAME = cfg.global_.wandb_project_name
    WANDB_RUN_NAME = cfg.global_.wandb_run_name
    WANDB_RUN_TAGS = cfg.global_.wandb_run_tags
    SAVE_MODEL_FLG = cfg.global_.save_model_flg
    SAVE_LIGHTING_LOGS_PATH = os.path.join(os.path.join(cfg.global_.save_path_root, WANDB_RUN_NAME))
    if not cfg.global_.save_model_path is None:
        SAVE_MODEL_PATH = cfg.global_.save_model_path
    else:
        SAVE_MODEL_PATH = os.path.join(cfg.global_.save_path_root, WANDB_RUN_NAME, "model.pth")
    os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)

    if cfg.global_.use_wandb:
        wandb.init(
            project=WANDB_PROJECT_NAME,
            name=WANDB_RUN_NAME,
            tags=WANDB_RUN_TAGS,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    set_seed(cfg.train.seed)

    train_dataset = HFImageDataset(split="train")
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=cfg.train.batch_size
    )
    validation_dataset = HFImageDataset(split="test")
    validation_loader = DataLoader(validation_dataset, batch_size=cfg.train.batch_size)
    
    if cfg.train.model_name == "resnet18-finetune":
        resnet18 = ResNet18ToFinetune()
    else:
        resnet18 = ResNet18()

    criterion = nn.CrossEntropyLoss()
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.train.num_classes)
    t_total = len(train_loader) * cfg.train.max_epoch

    model = ResnetClassifier(cfg.train.model_name, resnet18, criterion, metric, t_total, cfg.opt)
    logger = WandbLogger() if cfg.global_.use_wandb else None

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epoch,
        accelerator="gpu",
        devices=[0],
        logger=logger,
        val_check_interval=cfg.train.val_check_interval,
        default_root_dir=SAVE_LIGHTING_LOGS_PATH,
    )

    try:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=validation_loader,
        )

        if SAVE_MODEL_FLG:
            torch.save(model.clf.state_dict(), SAVE_MODEL_PATH)
    finally:
        if cfg.global_.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
