import torch
import torch.nn.functional as F
import numpy as np

SMOOTH = 1e-6
@torch.no_grad()
def calculate_IoU(pred, gt, thresh, size):
    size = size.cpu().numpy()
    pred[pred > thresh] = 1
    pred[pred <= thresh] = 0
    pred = F.interpolate(
                pred.unsqueeze(0).unsqueeze(0),
                size=(size[0], size[1])
            )
    gt = F.interpolate(
                gt.unsqueeze(0).unsqueeze(0),
                size=(size[0], size[1])
            )
    pred = pred.squeeze().cpu().numpy().astype(np.uint8)
    gt = gt.squeeze().cpu().numpy()
    IArea = (pred & (gt == 1.0)).astype(float).sum()
    OArea = (pred | (gt == 1.0)).astype(float).sum()
    IoU = (IArea + SMOOTH) / (OArea + SMOOTH)
    return IoU, IArea, OArea


@torch.no_grad()
def calculate_JF(pred, gt, thresh, size):
    size = size.cpu().numpy()
    pred[pred > thresh] = 1
    pred[pred <= thresh] = 0
    pred = F.interpolate(
                pred.unsqueeze(0).unsqueeze(0),
                size=(size[0], size[1])
            )
    gt = F.interpolate(
                gt.unsqueeze(0).unsqueeze(0),
                size=(size[0], size[1])
            )
    pred = pred.squeeze().cpu().numpy().astype(np.uint8)
    gt = gt.squeeze().cpu().numpy()
    IArea = (pred & (gt == 1.0)).astype(float).sum()
    OArea = (pred | (gt == 1.0)).astype(float).sum()
    IoU = (IArea + SMOOTH) / (OArea + SMOOTH)
    TP_FP = pred.sum()
    Prec = (IArea + SMOOTH) / (TP_FP + SMOOTH)
    TP_FN = gt.sum()
    Rec = (IArea + SMOOTH) / (TP_FN + SMOOTH)
    Contour = 2 * Prec * Rec / (Prec + Rec)
    return IoU, IArea, OArea, Contour

@torch.no_grad()
def compute_mask_IOU(masks, target, thresh=0.3):
    assert target.shape[-2:] == masks.shape[-2:]
    temp = (masks > thresh) * target
    intersection = temp.sum()
    union = (((masks > thresh) + target) - temp).sum()
    return intersection, union


@torch.no_grad()
def compute_batch_IOU(masks, target, thresh=0.3):
    assert target.shape[-2:] == masks.shape[-2:]
    temp = (masks > thresh) * target
    intersection = torch.sum(temp.flatten(1), dim=-1, keepdim=True)
    union = torch.sum(
        (((masks > thresh) + target) - temp).flatten(1), dim=-1, keepdim=True
    )
    return intersection, union
