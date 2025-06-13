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

from classes.bert_classifaer_for_tails import BertClassifaer
from classes.text_dataset import TextDatasetCoLa

MODEL_PATH = "roberta-large"
CHECKPOINT_FILE_PATH = "/Clipped-AdaGrad-and-Adam/ALBERT_fine_tuning/checkpoints/qnli_one_run_adam/qnli_final.pt"
SAVE_LOGS_PATH = "tails_qnli_adam_final.json"


@hydra.main(config_path="configs", config_name="config_qnli_adam", version_base="1.3")
def main(cfg: DictConfig):
    results = {}
    tokenizer = AutoTokenizer.from_pretrained(cfg.train.model_checkpoint)

    for task in cfg.data.tasks:
        actual_task = task if task not in ("mnli-mm",) else "mnli"
        os.makedirs(cfg.data.cache_dir, exist_ok=True)
        dataset = load_dataset("glue", actual_task, cache_dir=cfg.data.cache_dir)
        train_df = pd.DataFrame(dataset["train"][:])
        train_ds = TextDatasetCoLa(train_df, tokenizer)
        train_loader = DataLoader(train_ds, shuffle=True, batch_size=cfg.train.batch_size)

        if task == "cola":
            num_labels = 2
        elif task in ("rte",):
            num_labels = 2
        else:
            # snli, qnli, mnli, etc.
            num_labels = 3
        bert = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_labels, cache_dir=cfg.data.cache_dir)
        checkpoint = torch.load(CHECKPOINT_FILE_PATH, map_location="cpu")
        bert.load_state_dict(checkpoint["model_state_dict"])
        criterion = nn.CrossEntropyLoss()
        if num_labels == 2:
            metric = torchmetrics.MatthewsCorrCoef(task="binary")
        else:
            metric = torchmetrics.MatthewsCorrCoef(task="multiclass", num_classes=num_labels)
        t_total = len(train_loader) * cfg.train.max_epoch
        model = BertClassifaer(bert, criterion, metric, t_total, cfg.opt)

        trainer = pl.Trainer(max_epochs=1, accelerator="gpu", devices=[0,1,2,3], accumulate_grad_batches=len(train_loader), enable_checkpointing=False)
        trainer.fit(model, train_loader)
        true_grad = model.weights_grad

        norms = []
        for _ in range(100):
            bert2 = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_labels)
            bert2.load_state_dict(checkpoint["model_state_dict"])
            model2 = BertClassifaer(bert2, criterion, metric, t_total, cfg.opt)
            trainer2 = pl.Trainer(max_epochs=1, accelerator="gpu", devices=[0,1,2,3], enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
            ids = torch.randperm(len(train_df))[:cfg.train.batch_size]
            rand_ds = TextDatasetCoLa(train_df.iloc[ids.numpy(), :], tokenizer)
            rand_loader = DataLoader(rand_ds, batch_size=cfg.train.batch_size)
            trainer2.fit(model2, rand_loader)
            norms.append(torch.linalg.vector_norm(true_grad - model2.weights_grad).item())
        results[task] = norms

    with open(SAVE_LOGS_PATH, "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()