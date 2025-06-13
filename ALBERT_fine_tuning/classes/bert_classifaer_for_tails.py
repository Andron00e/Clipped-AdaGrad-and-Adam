import lightning.pytorch as pl
import torch

from ALBERT_fine_tuning.classes.optimizer import AdamClip, SGDClip

from ALBERT_fine_tuning.classes.utils import INDEP_LAYER_PER_NAME, get_per_layer_parameters


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

    def on_before_optimizer_step(self, optimizer):
        gradients = []
        for param in self.parameters():
            gradients.append(param.grad.flatten().clone())
            if not self.opt.do_training:
                param.grad.zero_() # do not want to update the model
        self.weights_grad = torch.cat(gradients, dim=0)

    def configure_optimizers(self):
        if self.opt.clipping == "layerwise":
            params = get_per_layer_parameters(
                self,
                INDEP_LAYER_PER_NAME[self.opt.model_name],
            )
            print("[INFO] Num of independent layers:", len(params))
        else:
            params = self.parameters()

        if self.opt.optimizer == "adam":
            optimizer = AdamClip(
                params,
                lr=self.opt.lr,
                clipping=self.opt.clipping,
                max_grad_norm=self.opt.max_grad_norm,
                weight_decay=self.opt.weight_decay,
            )
        elif self.opt.optimizer == "sgd":
            optimizer = SGDClip(
                params,
                lr=self.opt.lr,
                clipping=self.opt.clipping,
                max_grad_norm=self.opt.max_grad_norm,
                weight_decay=self.opt.weight_decay,
            )
        elif self.opt.optimizer == "adam-orig":
            optimizer = torch.optim.Adam(
                params,
                lr=self.opt.lr,
                weight_decay=self.opt.weight_decay,
            )
        else:
            raise RuntimeError(f"Wrong optimizer {self.opt.optimizer}")

        return {
            "optimizer": optimizer,
        }
