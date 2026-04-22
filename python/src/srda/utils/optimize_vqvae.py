import random
from logging import getLogger

import numpy as np
import torch
from src.srda.utils.utils import AverageMeter
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

logger = getLogger()


def optimize(
    *,
    mode: str,
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: Optimizer,
    device: str,
    epoch: int,
    **kwargs,
) -> float:
    if mode not in ["train", "valid", "test"]:
        raise NotImplementedError(f"{mode} is not supported.")

    if mode == "train":
        model.train()
    else:
        model.eval()

    np.random.seed(epoch)
    random.seed(epoch)
    loss_meter = AverageMeter()

    for Xs in dataloader:
        Xs = Xs[:, None, ...]
        Xs = Xs.to(device)

        if mode == "train":
            preds, embedding_loss, perplexity = model(Xs)
            loss = torch.mean((Xs - preds) ** 2) + embedding_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                preds, embedding_loss, perplexity = model(Xs)
                loss = torch.mean((Xs - preds) ** 2) + embedding_loss

        loss_meter.update(loss.item(), n=Xs.shape[0])

    logger.info(f"{mode}: avg loss = {loss_meter.avg:.8f}")

    return loss_meter.avg
