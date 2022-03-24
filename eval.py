import time
from datetime import datetime
from numpy.core.fromnumeric import transpose
import torch.nn.functional as F
import torch
import torch.nn as nn
from models.position_encoding import *

from utils.utils import print_
from utils.metrics import calculate_JF
import numpy as np
import cv2
import os.path as osp
import os
import torch.distributed as dist
from torchvision.utils import make_grid
from matplotlib import cm
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm

@torch.no_grad()
def visualize(args, pred, gt, img, diff, idx, orig_phrase, iou):
    # if not os.path.exists(osp.join('/mnt/data5/htr/CSTM/CSTM-CVPR21/visualize', orig_phrase, idx)):
    #     return
    _palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0,
                0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128, 191, 0, 128, 64, 128, 128, 191, 128, 128, 0, 64, 0,
                128, 64, 0, 0, 191, 0, 128, 191, 0, 0, 64, 128, 128, 64, 128, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25,
                25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33,
                34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42,
                42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51,
                51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59,
                60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68,
                68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77,
                77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85,
                86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94,
                94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102,
                102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109,
                109, 110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116,
                116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123,
                123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130,
                130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136, 137, 137,
                137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144,
                144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151,
                151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 157, 157, 157, 158, 158,
                158, 159, 159, 159, 160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165,
                165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171, 172, 172,
                172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179,
                179, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186,
                186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191, 192, 192, 192, 193, 193,
                193, 194, 194, 194, 195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200,
                200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206, 207, 207,
                207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214,
                214, 215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221,
                221, 222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228,
                228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235,
                235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242,
                242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249,
                249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255]
    # save_dir = osp.join('supplementary', args.model_name, orig_phrase, idx)
    save_dir = osp.join('supplementary', args.model_name+'_val', ('/').join(idx.split('/')[:2]))
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()*255
    img = img.cpu().numpy()*255
    img = img.transpose(1,2,0)
    img1 = img.copy()
    img1[:,:,0] = img[:,:,2]
    img1[:,:,1] = img[:,:,1]
    img1[:,:,2] = img[:,:,0]

    diff = diff.cpu().numpy()*255
    diff = diff.transpose(1,2,0)
    diff1 = diff.copy()
    diff1[:,:,0] = diff[:,:,2]
    diff1[:,:,1] = diff[:,:,1]
    diff1[:,:,2] = diff[:,:,0]

    img_diff = np.concatenate([img1, diff1], axis=1)
    cv2.imwrite(osp.join(save_dir, 'img_diff.png'), img_diff)

    color_id = int(idx.split('/')[-1])
    pred[pred > 0] = color_id + 1
    pred = Image.fromarray(pred).convert('P')
    pred.putpalette(_palette)
    pred = pred.convert('RGB')
    pred = np.array(pred)
    pred1 = pred.copy()
    pred1[:,:,0] = pred[:,:,2]
    pred1[:,:,1] = pred[:,:,1]
    pred1[:,:,2] = pred[:,:,0]
    cv2.imwrite(osp.join(save_dir, f'{color_id}.png'), pred1)



@torch.no_grad()
def vis_attn(args, pred, gt, img, diff, idx, orig_phrase, stage4, stage5, f4_1, f4_2, f5_1, f5_2):

    save_dir = osp.join('vis_diff_vai', args.model_name, idx+'_'+orig_phrase)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    pred = pred.cpu().numpy()*255
    gt = gt.cpu().numpy()*255
    img = img.cpu().numpy()*255
    img = img.transpose(1,2,0)
    img1 = img.copy()
    img1[:,:,0] = img[:,:,2]
    img1[:,:,1] = img[:,:,1]
    img1[:,:,2] = img[:,:,0]

    diff = diff.cpu().numpy()*255
    diff = diff.transpose(1,2,0)
    diff1 = diff.copy()
    diff1[:,:,0] = diff[:,:,2]
    diff1[:,:,1] = diff[:,:,1]
    diff1[:,:,2] = diff[:,:,0]

    img_diff = np.concatenate([img1, diff1], axis=1)
    cv2.imwrite(osp.join(save_dir, 'img_diff.png'), img_diff)

    stage4_max = stage4.max(dim=1, keepdim=True)[0]
    stage4_min = stage4.min(dim=1, keepdim=True)[0]
    stage4 = (stage4 - stage4_min) / (stage4_max - stage4_min)
    stage4 = stage4.reshape(25, 1, 20, 20)
    stage4 = F.interpolate(stage4, scale_factor=16, mode='bilinear', align_corners=True)
    stage4 = stage4.squeeze()
    stage4 = stage4.cpu().numpy()
    for i in range(len(orig_phrase.split())):
        mask = stage4[i][:, :, np.newaxis]
        mask = np.repeat(mask, 3, axis=2)
        mask[:, :, 0] = cm.jet(mask[:, :, 0])[:, :, 2]
        mask[:, :, 1] = cm.jet(mask[:, :, 1])[:, :, 1]
        mask[:, :, 2] = cm.jet(mask[:, :, 2])[:, :, 0]
        mask_diff = mask * 255 * 0.4 + diff1 * 0.6
        mask_img = mask * 255 * 0.4 + img1 * 0.6
        cv2.imwrite(osp.join(save_dir, f'stage4_diff_{orig_phrase.split()[i]}.png'), mask_diff)
        cv2.imwrite(osp.join(save_dir, f'stage4_img_{orig_phrase.split()[i]}.png'), mask_img)
    #
    #
    stage5_max = stage5.max(dim=1, keepdim=True)[0]
    stage5_min = stage5.min(dim=1, keepdim=True)[0]
    stage5 = (stage5 - stage5_min) / (stage5_max - stage5_min)
    stage5 = stage5.reshape(25, 1, 10, 10)
    stage5 = F.interpolate(stage5, scale_factor=32, mode='bilinear', align_corners=True)
    stage5 = stage5.squeeze()
    stage5 = stage5.cpu().numpy()
    for i in range(len(orig_phrase.split())):
        mask = stage5[i][:, :, np.newaxis]
        mask = np.repeat(mask, 3, axis=2)
        mask[:, :, 0] = cm.jet(mask[:, :, 0])[:, :, 2]
        mask[:, :, 1] = cm.jet(mask[:, :, 1])[:, :, 1]
        mask[:, :, 2] = cm.jet(mask[:, :, 2])[:, :, 0]
        mask_diff = mask * 255 * 0.4 + diff1 * 0.6
        mask_img = mask * 255 * 0.4 + img1 * 0.6
        cv2.imwrite(osp.join(save_dir, f'stage5_diff_{orig_phrase.split()[i]}.png'), mask_diff)
        cv2.imwrite(osp.join(save_dir, f'stage5_img_{orig_phrase.split()[i]}.png'), mask_img)

@torch.no_grad()
def vis_words_attn(args, pred, gt, img, diff, idx, orig_phrase, vis_dict, vis_id):
    save_dir = osp.join('words_vis', args.model_name, idx+'_'+orig_phrase)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    
    for stage in vis_dict.keys():
        stage_dict = vis_dict[stage]
        for type_vis in stage_dict.keys():
            type_dict = stage_dict[type_vis]
            fig, axes = plt.subplots(len(type_dict.keys()), figsize=(10, 4))
            fig.subplots_adjust(top=0.95, bottom=0.01, left=0.01, right=0.99)
            axes[0].set_title(stage+'_'+type_vis)
            for ax, part_vis in zip(axes, type_dict.keys()):
                part_info = type_dict[part_vis]
                vis_part = part_info[vis_id]
                vis_part = vis_part.sum(dim=0)
                tmp_max = vis_part.max()
                tmp_min = vis_part.min()
                vis_part_norm = (vis_part - tmp_min) / (tmp_max - tmp_min)
                vis_part_norm = vis_part_norm * 255
                vis_part_norm = vis_part_norm.int()
                norm_np = vis_part_norm.cpu().numpy()
                nonzero_ind = norm_np.nonzero()[0]
                norm_np = norm_np[nonzero_ind]
                norm_np = norm_np[np.newaxis, :]
                ax.imshow(norm_np, aspect='auto', cmap=plt.get_cmap('Blues'))
                ax.set_axis_off()
            plt.show()
            fig.savefig(osp.join(save_dir, f'{stage}_{type_vis}.png'))
    plt.cla()
    plt.close('all')

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
    prob = inputs.sigmoid()
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
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()

@torch.no_grad()
def save_annotations(args, ins_id, img_path, pred, size):
    save_path = img_path.replace('JPEGImages', f'results/{args.model_name}/{args.checkpoint}/Annotations')
    frame = img_path.split('/')[-1].replace('jpg', 'png')
    scene = '/'.join(save_path.split('/')[:-1])
    save_path = osp.join(scene, ins_id)
    if not osp.exists(save_path):
        os.makedirs(save_path)
    save_frame = osp.join(save_path, frame)
    pred = pred.unsqueeze(0).unsqueeze(0)
    pred = F.interpolate(pred, (size.numpy()[0], size.numpy()[1]), mode='bilinear', align_corners=True)
    pred = pred.squeeze()
    pred = pred.cpu().numpy()
    pred[pred > 0] = 1
    cv2.imwrite(save_frame, pred * 255)

@torch.no_grad()
def cal_fps(dataloader, model, args):
    # the first several iterations may be very slow so skip them
    num_warmup = 20
    pure_inf_time = 0
    fps = 0
    max_iter = 1000
    log_interval = 100

    # benchmark with 2000 image and take the average
    for i, batch in enumerate(dataloader):
        img = batch["image"].cuda(non_blocking=True)
        flow = batch['flow'].cuda(non_blocking=True)
        phrase = batch["phrase"].cuda(non_blocking=True)
        phrase_mask = batch['phrase_mask'].cuda(non_blocking=True)
        img_mask = torch.ones(
            args.batch_size, args.feature_dim * args.feature_dim, dtype=torch.int64
        ).cuda(non_blocking=True)

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            _ = model(img, flow, phrase, phrase_mask, img_mask)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break

@torch.no_grad()
def evaluate(
    val_loader,
    joint_model,
    epochId,
    args,
    eval_grounding=True,
):

    joint_model.eval()

    # bce_loss = nn.BCELoss()
    total_loss = 0
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5])).cuda()
    bce_loss_1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5])).cuda()
    bce_loss_2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5])).cuda()
    bce_loss_3 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5])).cuda()

    data_len = len(val_loader)

    MeanIoU, IArea, OArea, Overlap, MeanCont = [], [], [], [], []
    ind = 0

    for batch in tqdm(val_loader):
        orig_img = batch['orig_img']
        orig_flow = batch['orig_flow']
        img = batch["image"].cuda(non_blocking=True)
        flow = batch['flow'].cuda(non_blocking=True)
        phrase = batch["phrase"].cuda(non_blocking=True)
        orig_size = batch['orig_size']
        idx = batch['index']
        orig_phrase = batch['orig_phrase']
        orig_mask = batch['orig_mask'].cuda(non_blocking=True)
        batch_size = img.shape[0]
        phrase_mask = batch['phrase_mask'].cuda(non_blocking=True)
        img_mask = torch.ones(
                batch_size, args.feature_dim * args.feature_dim, dtype=torch.int64
            ).cuda(non_blocking=True)
        ins_id = batch['ins_id']
        img_path = batch['img_path']

        if not args.vis:
            # mask, _, _, _ = joint_model(img, flow, phrase, phrase_mask, img_mask)
            mask = joint_model(img, flow, phrase, phrase_mask, img_mask)

        else:
            mask, vis_dict = joint_model(img, flow, phrase, phrase_mask, img_mask, True)
        mask = mask.sigmoid()
        # total_loss += float(loss_ground.item())
        for i in range(batch_size):
            iou, iarea, oarea, cont = calculate_JF(mask[i], orig_mask[i], args.threshold, orig_size[i])
            if args.vis and iou >= 0.7:
                vis_words_attn(args, mask[i], orig_mask[i], orig_img[i], orig_flow[i],
                            '/'.join(img_path[i].split('/')[-2:])[:-4]+f'/{ins_id[i]}', orig_phrase[i],
                            vis_dict, i)
                # visualize(args, mask[i], orig_mask[i], orig_img[i], orig_flow[i], '/'.join(img_path[i].split('/')[-2:])[:-4]+f'/{ins_id[i]}', orig_phrase[i], iou)
            if args.task == 'ytvos' and args.save_result:
                save_annotations(args, ins_id[i], img_path[i], mask[i], orig_size[i])
            # visualize(mask[ground_indx][i], orig_mask[ground_indx][i], orig_img[i], str(int(idx[i])), orig_phrase[i], iou)
            MeanIoU.append(iou)
            IArea.append(iarea)
            OArea.append(oarea)
            Overlap.append(iou)
            MeanCont.append(cont)
            ind = ind + 1


    # val_loss = total_loss / data_len

    prec5, prec6, prec7, prec8, prec9 = np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1)), \
                                    np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1))
    for i in range(len(Overlap)):
        if Overlap[i] >= 0.5:
            prec5[i] = 1
        if Overlap[i] >= 0.6:
            prec6[i] = 1
        if Overlap[i] >= 0.7:
            prec7[i] = 1
        if Overlap[i] >= 0.8:
            prec8[i] = 1
        if Overlap[i] >= 0.9:
            prec9[i] = 1

    # maybe different with coco style as we could not get detailed response about the way to calculate it.
    # it is confuse to define precision and recall for me, if we follow the prior definition of precision.
    # anyone is welcome to pull request
    mAP_thres_list = list(range(50, 95+1, 5))
    mAP = []
    for i in range(len(mAP_thres_list)):
        tmp = np.zeros((len(Overlap), 1))
        for j in range(len(Overlap)):
            if Overlap[j] >= mAP_thres_list[i] / 100.0:
                tmp[j] = 1
        mAP.append(tmp.sum() / tmp.shape[0])

    mean_iou, mean_cont, overall_iou, precision5, precision6, precision7, precision8, precision9, precision_mAP = np.mean(np.array(MeanIoU)), np.mean(np.array(MeanCont)), np.array(IArea).sum() / np.array(OArea).sum(), \
           prec5.sum() / prec5.shape[0], prec6.sum() / prec6.shape[0], prec7.sum() / prec7.shape[0], \
           prec8.sum() / prec8.shape[0], prec9.sum() / prec9.shape[0], np.mean(np.array(mAP))
    timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")
    if args.dist and dist.get_rank() == 0:
        print(f'{timestamp} Validation: EpochId: {epochId+1:2d},\n'
              f'Precision@0.5 {precision5:.3f}, Precision@0.6 {precision6:.3f}, '
              f'Precision@0.7 {precision7:.3f}, Precision@0.8 {precision8:.3f}, Precision@0.9 {precision9:.3f},\n'
              f'mAP Precision @0.5:0.05:0.95 {precision_mAP:.3f},\n'
              f'Overall IoU {overall_iou:.3f}, Mean IoU (J) {mean_iou:.4f}, F {mean_cont:.4f}')


        print_("============================================================================================================================================\n")

    if args.task == 'ytvos':
        return (mean_cont + mean_iou) / 2
    else:
        return (overall_iou + mean_iou) / 2

