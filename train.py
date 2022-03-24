import os
import psutil
import gc
from time import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import print_
from utils.metrics import compute_mask_IOU
import torch.distributed as dist

def true_ce_loss(inputs, targets):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    ce_loss = ce_loss * targets
    return ce_loss.mean()

def _sigmoid(x):
    y = torch.clamp(torch.sigmoid(x), min=1e-4, max=1 - 1e-4)
    return y

def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = _sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = _sigmoid(inputs)
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()

def train(
    train_loader,
    joint_model,
    optimizer_a,
    epochId,
    args,
):

    pid = os.getpid()
    py = psutil.Process(pid)

    joint_model.train()


    total_loss = 0
    total_inter, total_union = 0, 0


    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5])).cuda()
    bce_loss_1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5])).cuda()
    bce_loss_2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5])).cuda()
    bce_loss_3 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5])).cuda()



    data_len = len(train_loader)

    epoch_start = time()
    if dist.get_rank() == 0:
        print_("\n=========================================================== Training Grounding Network ===================================================")
    for step, batch in enumerate(train_loader):
        iterId = step + (epochId * data_len) - 1
        with torch.no_grad():
            img = batch["image"].cuda(non_blocking=True)
            flow = batch['flow'].cuda(non_blocking=True)
            phrase = batch["phrase"].cuda(non_blocking=True)
            gt_mask = batch['orig_mask'].cuda(non_blocking=True)
            phrase_mask = batch['phrase_mask'].cuda(non_blocking=True)
            batch_size = img.shape[0]
            img_mask = torch.ones(
                batch_size, args.feature_dim * args.feature_dim, dtype=torch.int64
            ).cuda(non_blocking=True)
        start_time = time()
        # with torch.no_grad():
        pred = joint_model(img, flow, phrase, phrase_mask, img_mask)

        loss = 0.5 * bce_loss_3(pred, gt_mask) + 0.5 * dice_loss(pred, gt_mask)

        optimizer_a.zero_grad()
        loss.backward()
        optimizer_a.step()

        end_time = time()
        elapsed_time = end_time - start_time
        with torch.no_grad():
            inter, union = compute_mask_IOU(pred.sigmoid(), gt_mask, args.threshold)
        total_inter += inter.item()
        total_union += union.item()
        total_loss += float(loss.item())
        if iterId % 100 == 0 and step != 0:
            gc.collect()
            memoryUse = py.memory_info()[0] / 2.0 ** 20
            timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")
            curr_loss = total_loss / (step + 1)

            curr_IOU = total_inter / total_union
            lr = optimizer_a.param_groups[0]["lr"]
            if dist.get_rank() == 0:
                print_(
                    f"{timestamp} Epoch:[{epochId+1:2d}/{args.epochs:2d}] iter {iterId:6d} loss {curr_loss:.4f} IOU {curr_IOU:.4f} memory_use {memoryUse:.3f}MB lr {lr:.7f} elapsed {elapsed_time:.2f}"
                )

    epoch_end = time()
    epoch_time = epoch_end - epoch_start

    timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")

    train_loss = total_loss / data_len
    overall_IOU = total_inter / total_union

    if dist.get_rank() == 0:
        print_(
            f"{timestamp} FINISHED Epoch:{epochId+1:2d} loss {train_loss:.4f} overall_IOU {overall_IOU:.4f} elapsed {epoch_time:.2f}"
        )
        print_("============================================================================================================================================\n")
