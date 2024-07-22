from abc import abstractmethod
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging

from spikingjelly.activation_based.neuron import AdaptBaseNode
from spikingjelly.activation_based import surrogate

class AdaptiveIFNode(AdaptBaseNode):
  def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 v_rest: float = 0., w_rest: float = 0., tau_w: float = 2., a: float = 0., b: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False):
    super().__init__(1., v_reset, v_rest, w_rest, tau_w, a,b,surrogate_function, detach_reset, step_mode, backend, store_v_seq)

  def neuronal_charge(self, x: torch.Tensor):
    self.v = self.v + x - self.w


class AdaptiveLIFNode(AdaptBaseNode):
  def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 v_rest: float = 0., w_rest: float = 0., tau_w: float = 2., a: float = 0., b: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False, tau: float = 2.):
    super().__init__(1., v_reset, v_rest, w_rest, tau_w, a,b,surrogate_function, detach_reset, step_mode, backend, store_v_seq)
    self.tau = tau

  def neuronal_charge(self, x: torch.Tensor):
    self.v = self.v + (x - (self.v - self.v_reset)) / self.tau - self.w
