import lightning.pytorch as pl
import torch
from transformers import get_linear_schedule_with_warmup

from optimizer import AdamClip_Delayed_Etta


class BertClassifaer(pl.LightningModule):
    def __init__(self, model, criterion, metric, t_total, opt_dict):
        super().__init__()
        self.clf = model
        self.criterion = criterion
        self.metric = metric
        self.t_total = t_total
        self.opt = opt_dict
        self.weights_grad = None
        self.train_loss = []
        self.current_val_loss = []
        self.current_val_metric = []
        self.val_loss = []
        self.val_metric = []

    def training_step(self, batch, batch_idx):
        features, labels = batch
        model_ouput = self.clf(**features)
        loss = self.criterion(model_ouput.logits, labels)

        self.log("train/train_loss", loss, prog_bar=True)

        self.train_loss.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        model_ouput = self.clf(**features)
        loss = self.criterion(model_ouput.logits, labels)

        self.log("val/val_loss", loss, prog_bar=True)

        self.current_val_loss.append(loss.item())

        _, predicted = torch.max(model_ouput.logits, 1)
        metric = self.metric(predicted, labels)

        self.current_val_metric.append(metric.item())

        self.log("val/val_metric", metric.to(torch.float32), prog_bar=True)

    def on_validation_epoch_end(self):
        self.val_loss.append(self.current_val_loss)
        self.val_metric.append(self.current_val_metric)
        self.current_val_loss = []
        self.current_val_metric = []

    def test_step(self, batch, batch_idx):
        features, labels = batch
        model_ouput = self.clf(**features)
        loss = self.criterion(model_ouput.logits, labels)

        self.log("test/test_loss", loss, prog_bar=True)

        _, predicted = torch.max(model_ouput.logits, 1)
        metric = self.metric(predicted, labels)

        self.log("test/test_metric", metric.to(torch.float32), prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamClip_Delayed_Etta(
            self.parameters(),
            lr=self.opt.lr,
            clipping=self.opt.clipping,
            max_grad_norm=self.opt.max_grad_norm,
            weight_decay=self.opt.weight_decay,
            correct_bias=self.opt.correct_bias,
            betas=self.opt.betas,
            eps=self.opt.eps,
            exp_avg_sq_value=self.opt.exp_avg_sq_value,
            etta=self.opt.etta,
        )
        sheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * self.t_total),
            num_training_steps=self.t_total,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sheduler, "interval": "step"},
        }
