import logging
from typing import Any, Callable, Dict, Iterable, Optional, Union

try:
    from typing import override
except ImportError:
    override = lambda x: x


import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.utilities.exceptions import (
    MisconfigurationException,  # type: ignore #
)
from torch import nn
from torch.nn import Module
from torch.optim.optimizer import Optimizer

log = logging.getLogger(__name__)


def multiplicative(epoch: int) -> float:
    return 2.0


# Adapted from BackboneFinetuning in pytorch-lightning.callbacks
class NamedBackboneFinetuning(BaseFinetuning):

    def __init__(
        self,
        name: str = "imagenet",
        unfreeze_at_epoch: int = 10,
        lambda_func: Callable = multiplicative,
        initial_ratio_lr: float = 10e-2,
        initial_lr: Optional[float] = None,
        should_align: bool = True,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
        verbose: bool = False,
        rounding: int = 12,
    ) -> None:
        super().__init__()

        self.unfreeze_at_epoch: int = unfreeze_at_epoch
        self.lambda_func: Callable = lambda_func
        self.initial_ratio_lr: float = initial_ratio_lr
        self.initial_lr: Optional[float] = initial_lr
        self.should_align: bool = should_align
        self.initial_denom_lr: float = initial_denom_lr
        self.train_bn: bool = train_bn
        self.verbose: bool = verbose
        self.rounding: int = rounding
        self.previous_lr: Optional[float] = None
        self.name = name

    @property
    def state_key(self) -> str:
        return self._generate_state_key(name=self.name)

    @override
    def state_dict(self) -> Dict[str, Any]:
        return {
            "internal_optimizer_metadata": self._internal_optimizer_metadata,
            "previous_lr": self.previous_lr,
        }

    @override
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.previous_lr = state_dict["previous_lr"]
        super().load_state_dict(state_dict)

    @override
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Raises:
            MisconfigurationException:
                If LightningModule has no nn.Module `backbone` attribute.
        """
        if self.find_named_module(pl_module):
            return super().on_fit_start(trainer, pl_module)
        raise MisconfigurationException(f"The LightningModule should have a nn.Module `{self.name}` attribute")

    def find_named_module(self, pl_module: LightningModule) -> Iterable[nn.Module]:
        return [module for name, module in pl_module.named_modules() if name.endswith(self.name)]

    @override
    def unfreeze_and_add_param_group(
        self,
        modules: Union[Module, Iterable[Union[Module, Iterable]]],
        optimizer: Optimizer,
        lr: Optional[float] = None,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
    ) -> None:
        """Unfreezes a module and adds its parameters to an optimizer.

        Args:
            modules: A module or iterable of modules to unfreeze.
                Their parameters will be added to an optimizer as a new param group.
            optimizer: The provided optimizer will receive new parameters and will add them to
                `add_param_group`
            lr: Learning rate for the new param group.
            initial_denom_lr: If no lr is provided, the learning from the first param group will be used
                and divided by `initial_denom_lr`.
            train_bn: Whether to train the BatchNormalization layers.

        """
        BaseFinetuning.make_trainable(modules)
        params_lr = optimizer.param_groups[0]["lr"] if lr is None else float(lr)
        denom_lr = initial_denom_lr if lr is None else 1.0
        params = BaseFinetuning.filter_params(modules, train_bn=train_bn, requires_grad=True)
        params = BaseFinetuning.filter_on_optimizer(optimizer, params)
        if params:
            optimizer.add_param_group({"params": params, "lr": params_lr / denom_lr, "name": self.name})

    @override
    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(self.find_named_module(pl_module), train_bn=self.train_bn)

    def find_named_params_optimizer(self, optimizer: Optimizer) -> list[Dict[str, Any]]:
        param_groups = [param_group for param_group in optimizer.param_groups if param_group.get("name") == {self.name}]
        return param_groups

    @override
    def finetune_function(self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer) -> None:
        """Called when the epoch begins."""

        if self.unfreeze_at_epoch < 0:
            return
        elif epoch == self.unfreeze_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            initial_lr = self.initial_lr if self.initial_lr is not None else current_lr * self.initial_ratio_lr
            self.previous_lr = initial_lr
            self.unfreeze_and_add_param_group(
                self.find_named_module(pl_module),
                optimizer,
                initial_lr,
                train_bn=self.train_bn,
                initial_denom_lr=self.initial_denom_lr,
            )

            if self.verbose:
                log.info(f"Current lr: {round(current_lr, self.rounding)}, " f"Backbone lr: {round(initial_lr, self.rounding)}")

        elif epoch > self.unfreeze_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            next_current_lr = self.lambda_func(epoch + 1) * self.previous_lr
            next_current_lr = current_lr if (self.should_align and next_current_lr > current_lr) else next_current_lr

            for param_group in self.find_named_params_optimizer(optimizer):
                param_group["lr"] = next_current_lr
            self.previous_lr = next_current_lr
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.rounding)}, " f"Backbone lr: {round(next_current_lr, self.rounding)}"
                )
