import os
import json

import hydra
import lightning.pytorch as pl
import pandas as pd
import torch
import torchmetrics
from datasets import load_dataset
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data import HFImageDataset
from models import (
    ResNet18, 
    ResNet18ToFinetune
)
from tqdm import tqdm
import logging
from pathlib import Path

from classes import ResnetClassifier
from ALBERT_fine_tuning.classes.bert_classifaer_for_tails import BertClassifaer


@hydra.main(config_path="configs", config_name=None, version_base="1.3")
def main(cfg: DictConfig):
    torch.manual_seed(42)
    
    WANDB_PROJECT_NAME = cfg.global_.wandb_project_name
    WANDB_RUN_NAME = cfg.global_.wandb_run_name
    WANDB_RUN_TAGS = cfg.global_.wandb_run_tags
    SAVE_MODEL_FLG = cfg.global_.save_model_flg
    SAVE_MODEL_PATH = cfg.global_.save_model_path

    saved_epoch = 0
    if not SAVE_MODEL_PATH is None:
        name =  Path(SAVE_MODEL_PATH).name
        if name == "none": # very bad code
            SAVE_MODEL_PATH  = None
        else:
            saved_epoch = name.split("-")[0].split("=")[1]

    SAVE_NORM_PATH = os.path.join(cfg.global_.save_path_root, WANDB_RUN_NAME, f"norms_logs-{saved_epoch}.json")
    SAVE_LIGHTING_LOGS_PATH = os.path.join(cfg.global_.save_path_root, WANDB_RUN_NAME)
    # if not cfg.global_.save_model_path is None:
    # else:
        # SAVE_MODEL_PATH = os.path.join(cfg.global_.save_path_root, WANDB_RUN_NAME, "model.pth")

    train_dataset = HFImageDataset(split="train")
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=cfg.train.batch_size
    )


    criterion = nn.CrossEntropyLoss()
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.train.num_classes)
    t_total = len(train_loader) * cfg.train.max_epoch

    if cfg.opt.model_name == "resnet18-finetune":
        resnet18 = ResNet18ToFinetune()
    else:
        resnet18 = ResNet18()

    if not SAVE_MODEL_PATH is None:
        model = BertClassifaer.load_from_checkpoint(
            SAVE_MODEL_PATH,
            model=resnet18,
            criterion=criterion,
            metric=metric,
            t_total=t_total,
            opt_dict=cfg.opt,
        )

    else:
        model = BertClassifaer(
            model=resnet18,
            criterion=criterion,
            metric=metric,
            t_total=t_total,
            opt_dict=cfg.opt,
        )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=[0],
        accumulate_grad_batches=len(train_loader),
        enable_checkpointing=False,
        default_root_dir=SAVE_LIGHTING_LOGS_PATH,
    )

    trainer.fit(model=model, train_dataloaders=train_loader)
    true_grad = model.weights_grad
    stochastic_norms = []

    for _ in tqdm(range(1000)):
        if cfg.opt.model_name == "resnet18-finetune":
            resnet18 = ResNet18ToFinetune()
        else:
            resnet18 = ResNet18()
        if not SAVE_MODEL_PATH is None:
            model = BertClassifaer.load_from_checkpoint(
                SAVE_MODEL_PATH,
                model=resnet18,
                criterion=criterion,
                metric=metric,
                t_total=t_total,
                opt_dict=cfg.opt,
            )
        else:
            model = BertClassifaer(
                model=resnet18,
                criterion=criterion,
                metric=metric,
                t_total=t_total,
                opt_dict=cfg.opt,
            )

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=[0],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        ids = torch.randperm(len(train_dataset.dataset))[: cfg.train.batch_size].tolist()
        train_dataset.random_crop(ids)
        rand_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size)

        trainer.fit(model=model, train_dataloaders=rand_loader)
        stochastic_norm = torch.linalg.vector_norm(
            true_grad - model.weights_grad
        ).item()
        stochastic_norms.append(stochastic_norm)

    result = {"stochastic_norms": stochastic_norms}
    with open(SAVE_NORM_PATH, "w") as file:
        json.dump(result, file, indent=4)


if __name__ == "__main__":
    main()
