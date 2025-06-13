import json
import os
import gc

import hydra
import jsonlines
import lightning.pytorch as pl
import torch
import pandas as pd
import torchmetrics
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from classes import ResnetClassifier
from data import HFImageDataset
from models import ResNet18, ResNet18ToFinetune

# need to append root, it is done in ResnetClassifier file
from ALBERT_fine_tuning.functions.reproducibility import set_seed


@hydra.main(config_path="configs", config_name="config_cola", version_base="1.3")
def main(cfg: DictConfig):

    SEEDS = [0, 1, 12345, 42, 228]
    WANDB_PROJECT_NAME = cfg.global_.wandb_project_name
    WANDB_RUN_NAME = cfg.global_.wandb_run_name
    WANDB_RUN_TAGS = cfg.global_.wandb_run_tags
    SAVE_MODEL_FLG = cfg.global_.save_model_flg
    SAVE_LIGHTING_LOGS_PATH = os.path.join(os.path.join(cfg.global_.save_path_root, WANDB_RUN_NAME))
    SAVE_LOGS_PATH = os.path.join(cfg.global_.save_path_root, WANDB_RUN_NAME, "logs.json")
    if not cfg.global_.save_model_path is None:
        SAVE_MODEL_PATH = cfg.global_.save_model_path
    else:
        SAVE_MODEL_PATH = os.path.join(cfg.global_.save_path_root, WANDB_RUN_NAME, "model.pth")
    os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)

    for seed in SEEDS:
        set_seed(seed)

        train_dataset = HFImageDataset(split="train")
        train_loader = DataLoader(
            train_dataset, shuffle=True, batch_size=cfg.train.batch_size
        )
        validation_dataset = HFImageDataset(split="test")
        validation_loader = DataLoader(validation_dataset, batch_size=cfg.train.batch_size)

        if cfg.opt.model_name == "resnet18-finetune":
            resnet18 = ResNet18ToFinetune()
        else:
            resnet18 = ResNet18()

        criterion = nn.CrossEntropyLoss()
        metric = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.train.num_classes)
        t_total = len(train_loader) * cfg.train.max_epoch

        model = ResnetClassifier(cfg.opt.model_name, resnet18, criterion, metric, t_total, cfg.opt)

        trainer = pl.Trainer(
            max_epochs=cfg.train.max_epoch,
            accelerator="gpu",
            devices=[0],    
            check_val_every_n_epoch=1,
            default_root_dir=f"{SAVE_LIGHTING_LOGS_PATH}-{seed}",
        )

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=validation_loader,
        )

        with jsonlines.open(SAVE_LOGS_PATH, mode="a") as writer:
            d = {
                "config": OmegaConf.to_container(cfg),
                "model": {
                    "train_loss": model.train_loss,
                    "val_loss": model.val_loss,
                    "val_metric": model.val_metric,
                },
            }
            writer.write(d)

        # free all stuff
        del model, trainer, resnet18
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
