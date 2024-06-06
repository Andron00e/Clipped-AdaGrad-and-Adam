import lightning.pytorch as pl
import torch

from classes.optimizer import AdamClip


class BertClassifaer(pl.LightningModule):
    def __init__(self, model, criterion, metric, t_total, opt_dict):
        super().__init__()
        self.clf = model
        self.criterion = criterion
        self.metric = metric
        self.t_total = t_total
        self.opt = opt_dict
        self.weights_grad = None

    def training_step(self, batch, batch_idx):
        features, labels = batch
        model_ouput = self.clf(**features)
        loss = self.criterion(model_ouput.logits, labels)

        return loss

    def on_before_optimizer_step(self, optimizer):
        gradients = []
        for param in self.parameters():
            gradients.append(param.grad.flatten())
        self.weights_grad = torch.cat(gradients, dim=0)

    def configure_optimizers(self):
        optimizer = AdamClip(
            self.parameters(),
            lr=self.opt.lr,
            clipping=self.opt.clipping,
            max_grad_norm=self.opt.max_grad_norm,
            weight_decay=self.opt.weight_decay,
        )
        return {
            "optimizer": optimizer,
        }
