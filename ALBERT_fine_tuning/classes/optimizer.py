import math

import torch
from torch.optim import Optimizer
import itertools

class AdamClip(Optimizer):
    """
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
        clipping (str): "no", "local", "elementwise".  Default: "no"
        max_grad_norm (float): value to which we clip the gradient. Default: 1.0
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.0,
        correct_bias=True,
        clipping="no",
        max_grad_norm=1.0,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
            clipping=clipping,
            max_grad_norm=max_grad_norm,
        )

        self._max_grad_norm = max_grad_norm
        self._clipping = clipping

        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        if self._clipping == "global":
            all_params = [p for el in self.param_groups for p in el["params"]]
            torch.nn.utils.clip_grad_norm_(all_params, self._max_grad_norm)

        for group in self.param_groups:
            if group["clipping"] == "layerwise":
                torch.nn.utils.clip_grad_norm_(group["params"], group["max_grad_norm"])
            for p in group["params"]:
                if p.grad is None:
                    continue

                if group["clipping"] == "local":
                    torch.nn.utils.clip_grad_norm_(p, group["max_grad_norm"])
                elif group["clipping"] == "elementwise":
                    torch.nn.utils.clip_grad_value_(p, group["max_grad_norm"])
                
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]

                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                p.data.addcdiv_(tensor1=exp_avg, tensor2=denom, value=-step_size)

                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss


class SGDClip(Optimizer):
    """
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
        clipping (str): "no", "local", "elementwise".  Default: "no"
        max_grad_norm (float): value to which we clip the gradient. Default: 1.0
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.0,
        clipping="no",
        max_grad_norm=1.0,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
    
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            clipping=clipping,
            max_grad_norm=max_grad_norm,
        )

        self._max_grad_norm = max_grad_norm
        self._clipping = clipping

        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        if self._clipping == "global":
            all_params = [p for el in self.param_groups for p in el["params"]]
            torch.nn.utils.clip_grad_norm_(all_params, self._max_grad_norm)

        for group in self.param_groups:
            if group["clipping"] == "layerwise":
                torch.nn.utils.clip_grad_norm_(group["params"], group["max_grad_norm"])
            for p in group["params"]:
                if p.grad is None:
                    continue

                if group["clipping"] == "local":
                    torch.nn.utils.clip_grad_norm_(p, group["max_grad_norm"])
                elif group["clipping"] == "elementwise":
                    torch.nn.utils.clip_grad_value_(p, group["max_grad_norm"])
                
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                state["step"] += 1

                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
                
                p.data.add_(grad, alpha=-group["lr"])

        return loss
