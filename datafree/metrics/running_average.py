import numpy as np
import torch
from .stream_metrics import Metric

__all__=['Accuracy', 'TopkAccuracy']

class RunningLoss(Metric):
    def __init__(self, loss_fn, is_batch_average=False):
        self.reset()
        self.loss_fn = loss_fn
        self.is_batch_average = is_batch_average

    @torch.no_grad()
    def update(self, outputs, targets):
        self._accum_loss += self.loss_fn(outputs, targets)
        if self.is_batch_average:
            self._cnt += 1
        else:
            self._cnt += len(outputs)

    def get_results(self):
        return (self._accum_loss / self._cnt).detach().cpu()
    
    def reset(self):
        self._accum_loss = self._cnt = 0.0
