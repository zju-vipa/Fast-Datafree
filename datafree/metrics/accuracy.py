import numpy as np
import torch
from .stream_metrics import Metric
from typing import Callable

__all__=['Accuracy', 'TopkAccuracy']

class Accuracy(Metric):
    def __init__(self):
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs = outputs.max(1)[1]
        self._correct += ( outputs.view(-1)==targets.view(-1) ).sum()
        self._cnt += torch.numel( targets )

    def get_results(self):
        return (self._correct / self._cnt * 100.).detach().cpu()
    
    def reset(self):
        self._correct = self._cnt = 0.0


class TopkAccuracy(Metric):
    def __init__(self, topk=(1, 5)):
        self._topk = topk
        self.reset()
    
    @torch.no_grad()
    def update(self, outputs, targets):
        for k in self._topk:
            _, topk_outputs = outputs.topk(k, dim=1, largest=True, sorted=True)
            correct = topk_outputs.eq( targets.view(-1, 1).expand_as(topk_outputs) )
            self._correct[k] += correct[:, :k].view(-1).float().sum(0).item()
        self._cnt += len(targets)

    def get_results(self):
        return tuple( self._correct[k] / self._cnt * 100. for k in self._topk )

    def reset(self):
        self._correct = {k: 0 for k in self._topk}
        self._cnt = 0.0