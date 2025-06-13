import json
import hydra
import jsonlines
import lightning.pytorch as pl
import pandas as pd
import torch
import torchmetrics
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from classes.bert_classifaer import BertClassifaer
from functions.reproducibility import set_seed

PATH_TO_SEEDS = "random_seed.json"
MODEL_PATH    = "roberta-large"
SAVE_LOGS_PATH = "qnli_adam_logs.jsonl"


class HFTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        return item, self.labels[idx]


@hydra.main(config_path="configs", config_name="config_qnli_adam", version_base="1.3")
def main(cfg: DictConfig):
    seeds     = json.load(open(PATH_TO_SEEDS))
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.train.model_checkpoint,
        cache_dir=cfg.data.cache_dir,
        return_token_type_ids=True,
    )

    for seed in seeds:
        set_seed(seed)
        for task in cfg.data.tasks:
            print(f"\n>>> Processing task: {task}")
            actual = "mnli" if task == "mnli-mm" else task
            ds     = load_dataset("glue", actual, cache_dir=cfg.data.cache_dir)

            val_key = "validation" if task in ("cola", "rte", "qnli") else "validation_matched"

            train_df = pd.DataFrame(ds["train"][:])
            val_df   = pd.DataFrame(ds[val_key][:])

            if task == "cola":
                enc_train = tokenizer(
                    train_df["sentence"].tolist(),
                    padding="max_length",
                    truncation=True,
                    max_length=cfg.train.max_length,
                    return_tensors="pt",
                )
            elif task == "rte":
                enc_train = tokenizer(
                    train_df["sentence1"].tolist(),
                    train_df["sentence2"].tolist(),
                    padding="max_length",
                    truncation=True,
                    max_length=cfg.train.max_length,
                    return_tensors="pt",
                )
            elif task == "qnli":
                enc_train = tokenizer(
                    train_df["question"].tolist(),
                    train_df["sentence"].tolist(),
                    padding="max_length",
                    truncation=True,
                    max_length=cfg.train.max_length,
                    return_tensors="pt",
                )
            else:
                raise ValueError(f"Unsupported task: {task}")

            if task == "cola":
                enc_val = tokenizer(
                    val_df["sentence"].tolist(),
                    padding="max_length",
                    truncation=True,
                    max_length=cfg.train.max_length,
                    return_tensors="pt",
                )
            elif task == "rte":
                enc_val = tokenizer(
                    val_df["sentence1"].tolist(),
                    val_df["sentence2"].tolist(),
                    padding="max_length",
                    truncation=True,
                    max_length=cfg.train.max_length,
                    return_tensors="pt",
                )
            elif task == "qnli":
                enc_val = tokenizer(
                    val_df["question"].tolist(),
                    val_df["sentence"].tolist(),
                    padding="max_length",
                    truncation=True,
                    max_length=cfg.train.max_length,
                    return_tensors="pt",
                )

            for enc in (enc_train, enc_val):
                if "token_type_ids" not in enc:
                    bsz, seq_len = enc["input_ids"].shape
                    enc["token_type_ids"] = torch.zeros((bsz, seq_len), dtype=torch.long)

            train_ds = HFTextDataset(enc_train, torch.tensor(train_df["label"].tolist()))
            val_ds   = HFTextDataset(enc_val,   torch.tensor(val_df["label"].tolist()))
            train_loader = DataLoader(train_ds, shuffle=True, batch_size=cfg.train.batch_size)
            val_loader   = DataLoader(val_ds,   batch_size=cfg.train.batch_size)

            num_labels = 2
            bert = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH, num_labels=num_labels, cache_dir=cfg.data.cache_dir
            )
            criterion = nn.CrossEntropyLoss()
            metric = torchmetrics.MatthewsCorrCoef(task="binary")

            t_total = len(train_loader) * cfg.train.max_epoch
            model   = BertClassifaer(bert, criterion, metric, t_total, cfg.opt)

            trainer = pl.Trainer(
                max_epochs=cfg.train.max_epoch,
                accelerator="gpu",
                devices=[0,1,2,3],
                val_check_interval=cfg.train.val_check_interval,
            )
            trainer.fit(model, train_loader, val_loader)

            with jsonlines.open(SAVE_LOGS_PATH, mode="a") as writer:
                writer.write({
                    "seed": seed,
                    "task": task,
                    "train_loss": model.train_loss,
                    "val_loss":   model.val_loss,
                    "val_metric": model.val_metric,
                })


if __name__ == "__main__":
    main()
