import sys
from pathlib import Path

curr_path = Path(__file__)
# print("take from:", str(curr_path.parent.parent.parent / "ALBERT_fine_tuning/classes"))
sys.path.append(str(curr_path.parent.parent.parent / "ALBERT_fine_tuning/classes"))
from bert_classifaer import BertClassifaer
import torch


class ResnetClassifier(BertClassifaer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_optimizers(self):
        # optimizer_and_scheduler = BertClassifaer.configure_optimizers(self)
        # optimizer_and_scheduler["lr_scheduler"] = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer_and_scheduler["optimizer"], 
        #     T_max=200
        # )
        
        # return optimizer_and_scheduler
        optimizer = torch.optim.SGD(self.parameters(), lr=self.opt.lr,
                      momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.t_total)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
