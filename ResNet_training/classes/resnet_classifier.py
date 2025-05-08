import sys
from pathlib import Path

curr_path = Path(__file__)
# print("take from:", str(curr_path.parent.parent.parent / "ALBERT_fine_tuning/classes"))
sys.path.append(str(curr_path.parent.parent.parent / "ALBERT_fine_tuning/classes"))
from bert_classifaer import BertClassifaer
import torch


class ResnetClassifier(BertClassifaer):
    def __init__(self, model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    def configure_optimizers(self):
        if self.model_name == "resnet18-finetune":
            optimizer_and_scheduler = BertClassifaer.configure_optimizers(self)
            return optimizer_and_scheduler
        else:
            # return optimizer_and_scheduler
            optimizer = torch.optim.SGD(self.parameters(), lr=self.opt.lr,
                        momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.t_total)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
