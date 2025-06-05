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

from classes.bert_classifaer_for_tails import BertClassifaer
from classes.text_dataset import TextDatasetCoLa
import os

@hydra.main(config_path="configs", config_name="config_cola", version_base="1.3")
def main(cfg: DictConfig):
    WANDB_RUN_NAME = cfg.global_.wandb_run_name
    SAVE_NORM_PATH = os.path.join(cfg.global_.save_path_root, WANDB_RUN_NAME, "norms_logs.json")
    SAVE_LIGHTING_LOGS_PATH = os.path.join(os.path.join(cfg.global_.save_path_root, WANDB_RUN_NAME))
    if not cfg.global_.save_model_path is None:
        SAVE_MODEL_PATH = cfg.global_.save_model_path
    else:
        SAVE_MODEL_PATH = os.path.join(cfg.global_.save_path_root, WANDB_RUN_NAME)

    actual_task = "mnli" if cfg.data.task == "mnli-mm" else cfg.data.task
    dataset = load_dataset("glue", actual_task)
    num_labels = (
        3 if cfg.data.task.startswith("mnli") else 1 if cfg.data.task == "stsb" else 2
    )
    tokenizer = AutoTokenizer.from_pretrained(SAVE_MODEL_PATH)
    train_data = pd.DataFrame(dataset["train"][:])

    train_dataset = TextDatasetCoLa(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=cfg.train.batch_size
    )
    bert = AutoModelForSequenceClassification.from_pretrained(
        SAVE_MODEL_PATH, num_labels=num_labels
    )

    criterion = nn.CrossEntropyLoss()
    metric = torchmetrics.MatthewsCorrCoef(task="binary")
    t_total = len(train_loader) * cfg.train.max_epoch

    model = BertClassifaer(
        model=bert,
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

    for _ in range(1000):
        bert = AutoModelForSequenceClassification.from_pretrained(
            cfg.train.model_checkpoint, num_labels=num_labels
        )

        model = BertClassifaer(
            model=bert,
            criterion=criterion,
            metric=metric,
            t_total=t_total,
            opt_dict=cfg.opt,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=[0],
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        ids = torch.randperm(len(train_data))[: cfg.train.batch_size]
        rand_dataset = TextDatasetCoLa(train_data.iloc[ids.numpy(), :], tokenizer)

        rand_loader = DataLoader(rand_dataset, batch_size=cfg.train.batch_size)

        trainer.fit(model=model, train_dataloaders=rand_loader)
        stochastic_norm = torch.linalg.vector_norm(
            true_grad - model.weights_grad
        ).item()
        stochastic_norms.append(stochastic_norm)

    result = {"stochastic_norms": stochastic_norms}
    with open(
        SAVE_NORM_PATH,
        "w",
    ) as file:
        json.dump(result, file)


if __name__ == "__main__":
    main()
