import torch
from torch.autograd import Variable


def calculate_accuracy(predictions, targets):
    if isinstance(predictions, Variable):
        predictions = predictions.data
    if isinstance(targets, Variable):
        targets = targets.data

    predicted_classes = predictions.topk(1)[1].squeeze(1)
    return [accuracy.item() for accuracy in torch.eq(predicted_classes, targets)]
