"""PyTorch module API for superquantile-based reducers.

.. moduleauthor:: Krishna Pillutla
"""
import torch
from .functional import (
    reduce_mean, reduce_superquantile, reduce_superquantile_smooth
)

class MeanReducer(torch.nn.Module):
    """Reduce a batch of values by their mean.

    :param batch: Tensor of values to reduce. Reshaped into a single dimension.
    :type batch: torch.Tensor
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return reduce_mean(batch)

class SuperquantileReducer(torch.nn.Module):
    r"""Reduce a batch of values by their superquantile (a.k.a. Conditional Value at Risk, CVaR).

    It is given by the value of the linear program

    .. math:: 
        
        \min \quad & \,\, \langle q, \ell\rangle \\
        \text{s.t. } \quad &  0 \le q \le \frac{1}{\theta n}, \\ 
            & \sum_{i=1}^n q_i = 1,

    where :math:`\ell` represents the input `batch` to the `forward` method,
    :math:`n` is the dimension of :math:`\ell` (and also of :math:`q`) and 
    :math:`\theta` is `superquantile_tail_fraction` and :math:`\nu` is `smoothing_coefficient`.

    :param batch: Tensor of values to reduce. Reshaped into a single dimension.
    :type batch: torch.Tensor
    :param superquantile_tail_fraction: What fraction of the tail to average over for the superquantile. 
        If 1.0, the function returns `batch.mean()` and if 0.0, the function returns `batch.max()`.
    :type superquantile_tail_fraction: float, default is 0.5

    Examples::

        criterion = torch.nn.CrossEntropyLoss(reduction='none')  # Note: must set `reduction='none'`
        reducer = SuperquantileReducer(superquantile_tail_fraction=0.6)  # define the reducer

        # Training loop
        for x, y in dataloader:
            y_hat = model(x)
            batch_losses = criterion(y_hat, y)  # shape: (batch_size,)
            loss = reducer(batch_losses)  # Additional line to use the superquantile reducer
            loss.backward()  # Proceed as usual from here
            ...
    """
    def __init__(self, superquantile_tail_fraction=0.5):
        super().__init__()
        self.superquantile_tail_fraction = superquantile_tail_fraction

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return reduce_superquantile(batch, self.superquantile_tail_fraction)

class SuperquantileSmoothReducer(torch.nn.Module):
    r"""Reduce a batch of values by a L2 smoothing of their superquantile (a.k.a. Conditional Value at Risk, CVaR).

    It is given by the value of the quadratic program
 
    .. math:: 
        
        \min  \quad & \, \, \langle q, \ell \rangle  - \frac{\nu}{2n} \big\| q - \frac{\mathbf{1}_n}{n} \big\|_2^2 \\
        \text{s.t. } \quad &  0 \le q \le \frac{1}{\theta n},  
            \\ & \sum_{i=1}^n q_i = 1,

    where :math:`\ell` represents `batch`,
    :math:`n` is the dimension of :math:`\ell` (and also of :math:`q`) and 
    :math:`\theta` is `superquantile_tail_fraction` and :math:`\nu` is `smoothing_coefficient`.

    :param batch: Tensor of values to reduce. Reshaped into a single dimension.
    :type batch: torch.Tensor
    :param superquantile_tail_fraction: What fraction of the tail to average over for the superquantile. 
        If 1.0, the function returns `batch.mean()` and if 0.0, the function returns `batch.max()`.
    :type superquantile_tail_fraction: float, default is 0.5
    :param smoothing_coefficient: How much to smooth by? 
    :type smoothing_coefficient: float, default=1.0 

    Examples::

        criterion = torch.nn.CrossEntropyLoss(reduction='none')  # Note: must set `reduction='none'`
        reducer = SuperquantileSmoothReducer(  # define the reducer
            superquantile_tail_fraction=0.6, smoothing_coefficient=1.0
        ) 

        # Training loop
        for x, y in dataloader:
            y_hat = model(x)
            batch_losses = criterion(y_hat, y)  # shape: (batch_size,)
            loss = reducer(batch_losses)  # Additional line to use the superquantile reducer
            loss.backward()  # Proceed as usual from here
            ...
    """
    def __init__(self, superquantile_tail_fraction, smoothing_coefficient):
        super().__init__()
        self.superquantile_tail_fraction = superquantile_tail_fraction
        self.smoothing_coefficient = smoothing_coefficient

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return reduce_superquantile_smooth(
            batch, self.superquantile_tail_fraction, self.smoothing_coefficient
        )