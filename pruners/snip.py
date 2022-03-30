import torch
import numpy as np
from utils import dataflow

from .pruner import Pruner

# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        ## imagenet
        # for batch_idx, (data, target) in enumerate(dataloader):
        #     data, target = data.to(device), target.to(device)
        #     output = model(data)
        #     loss(output, target).backward()

        ## keypoint coco
        # data_iterator = iter(dataloader)
        # for batch_idx, data in enumerate(data_iterator):
        #     input, target, target_weight, meta = data
        #     input, target, target_weight = input.to(device), target.to(device), target_weight.to(device)
        #     outputs = model(input)
        #     if isinstance(outputs, list):
        #         _loss = loss(outputs[0], target, target_weight)
        #         for output in outputs[1:]:
        #             _loss += loss(output, target, target_weight)
        #     else:
        #         output = outputs
        #         _loss = loss(output, target, target_weight)
        #     _loss.backward()

        ## ade20k / cityscapes
        data_iterator = iter(dataloader)
        for batch_idx, data in enumerate(data_iterator):
            if batch_idx > 100:
                break
            input, target = data
            input = input.to(device)
            target = target.cuda(non_blocking=True)
            output = model(input)
            _loss, acc = loss(output, target)
            _loss.backward()

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)