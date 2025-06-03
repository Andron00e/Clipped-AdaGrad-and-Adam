import hydra
import lightning.pytorch as pl
import pandas as pd
import torchmetrics
import wandb
from classes.bert_classifaer import BertClassifaer
from classes.text_dataset import TextDatasetRTE
from datasets import load_dataset
from functions.reproducibility import set_seed
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os


@hydra.main(config_path="configs", config_name="config_rte", version_base="1.3")
def main(cfg: DictConfig):
    WANDB_PROJECT_NAME = cfg.global_.wandb_project_name
    WANDB_RUN_NAME = cfg.global_.wandb_run_name
    WANDB_RUN_TAGS = cfg.global_.wandb_run_tags
    MODEL_PATH = cfg.global_.model_name_or_path

    SAVE_MODEL_FLG = cfg.global_.save_model_flg
    if not cfg.global_.save_model_path is None:
        SAVE_MODEL_PATH = cfg.global_.save_model_path
    else:
        SAVE_MODEL_PATH = os.path.join(cfg.global_.save_path_root, WANDB_RUN_NAME)
    os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)
    
    wandb.init(
        project=WANDB_PROJECT_NAME,
        name=WANDB_RUN_NAME,
        tags=WANDB_RUN_TAGS,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    set_seed(cfg.train.seed)

    actual_task = "mnli" if cfg.data.task == "mnli-mm" else cfg.data.task
    dataset = load_dataset("glue", actual_task)
    print("Dataset:", dataset)

    train_data = pd.DataFrame(dataset["train"][:])
    validation_data = pd.DataFrame(dataset["validation"][:])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    train_dataset = TextDatasetRTE(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=cfg.train.batch_size
    )
    validation_dataset = TextDatasetRTE(validation_data, tokenizer)
    validation_loader = DataLoader(validation_dataset, batch_size=cfg.train.batch_size)

    num_labels = (
        3 if cfg.data.task.startswith("mnli") else 1 if cfg.data.task == "stsb" else 2
    )

    bert = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=num_labels
    )

    criterion = nn.CrossEntropyLoss()
    metric = torchmetrics.Accuracy(task="binary")
    t_total = len(train_loader) * cfg.train.max_epoch

    model = BertClassifaer(bert, criterion, metric, t_total, cfg.opt)

    # logger = WandbLogger() if cfg.global_.use_wandb else None
    logger = None

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epoch,
        accelerator="gpu",
        devices=[0],
        logger=logger,
        val_check_interval=cfg.train.val_check_interval,
    )

    try:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=validation_loader,
        )

        if SAVE_MODEL_FLG:
            model.clf.save_pretrained(SAVE_MODEL_PATH)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
