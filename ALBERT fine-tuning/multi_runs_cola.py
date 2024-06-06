import json

import hydra
import jsonlines
import lightning.pytorch as pl
import pandas as pd
import torchmetrics
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from classes.bert_classifaer import BertClassifaer
from classes.text_dataset import TextDatasetCoLa
from functions.reproducibility import set_seed

PATH_TO_SEEDS = "random_seed.json"
MODEL_PATH = "albert-base-v2"
SAVE_LOGS_PATH = "save_logs_path.jsonl"


@hydra.main(config_path="configs", config_name="config_cola", version_base="1.3")
def main(cfg: DictConfig):
    seeds = []
    with open(PATH_TO_SEEDS, "r") as file:
        seeds = json.load(file)

    for seed in seeds:
        set_seed(seed)

        actual_task = "mnli" if cfg.data.task == "mnli-mm" else cfg.data.task
        dataset = load_dataset("glue", actual_task)

        train_data = pd.DataFrame(dataset["train"][:])
        validation_data = pd.DataFrame(dataset["validation"][:])

        tokenizer = AutoTokenizer.from_pretrained(cfg.train.model_checkpoint)

        train_dataset = TextDatasetCoLa(train_data, tokenizer)
        train_loader = DataLoader(
            train_dataset, shuffle=True, batch_size=cfg.train.batch_size
        )
        validation_dataset = TextDatasetCoLa(validation_data, tokenizer)
        validation_loader = DataLoader(
            validation_dataset, batch_size=cfg.train.batch_size
        )

        num_labels = (
            3
            if cfg.data.task.startswith("mnli")
            else 1
            if cfg.data.task == "stsb"
            else 2
        )

        bert = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH, num_labels=num_labels
        )

        criterion = nn.CrossEntropyLoss()
        metric = torchmetrics.MatthewsCorrCoef(task="binary")
        t_total = len(train_loader) * cfg.train.max_epoch

        model = BertClassifaer(bert, criterion, metric, t_total, cfg.opt)

        trainer = pl.Trainer(
            max_epochs=cfg.train.max_epoch,
            accelerator="gpu",
            devices=[0],
            val_check_interval=cfg.train.val_check_interval,
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


if __name__ == "__main__":
    main()
